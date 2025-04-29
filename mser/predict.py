import os
import sys
from io import BufferedReader
from typing import List

import joblib
import numpy as np
import torch
import yaml

from loguru import logger
from yeaudio.audio import AudioSegment
from mser import SUPPORT_EMOTION2VEC_MODEL
from mser.data_utils.featurizer import AudioFeaturizer
from mser.models import build_model
from mser.utils.utils import dict_to_object, print_arguments, convert_string_based_on_type


class MSERPredictor:
    def __init__(self,
                 configs,
                 use_ms_model=None,
                 model_path='models/BiLSTM_Emotion2Vec/best_model/',
                 use_gpu=True,
                 overwrites=None,
                 log_level="info"):
        """Speech Emotion Recognition prediction utility class

        :param configs: Path to configuration file, or model name; if it's a model name, the default config is used
        :param use_ms_model: Use a ModelScope publicly available Emotion2Vec model
        :param model_path: Path to the exported model directory
        :param use_gpu: Whether to use GPU for prediction
        :param overwrites: Overwrite config file parameters, e.g., "train_conf.max_epoch=100", separated by commas
        :param log_level: Logging level; options are "debug", "info", "warning", "error"
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU is not available'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")

        self.log_level = log_level.upper()
        logger.remove()
        logger.add(sink=sys.stdout, level=self.log_level)

        self.use_ms_model = use_ms_model
        if use_ms_model is not None:
            # Using ModelScope public models
            assert use_ms_model in SUPPORT_EMOTION2VEC_MODEL, f'Model not supported: {use_ms_model}'
            from mser.utils.emotion2vec_predict import Emotion2vecPredict
            self.predictor = Emotion2vecPredict(use_ms_model, revision=None, use_gpu=use_gpu)
            return

        # Load config file
        if isinstance(configs, str):
            absolute_path = os.path.dirname(__file__)
            config_path = os.path.join(absolute_path, f"configs/{configs}.yml")
            configs = config_path if os.path.exists(config_path) else configs
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.configs = dict_to_object(configs)

        # Overwrite config parameters if specified
        if overwrites:
            overwrites = overwrites.split(",")
            for overwrite in overwrites:
                keys, value = overwrite.strip().split("=")
                attrs = keys.split('.')
                current_level = self.configs
                for attr in attrs[:-1]:
                    current_level = getattr(current_level, attr)
                before_value = getattr(current_level, attrs[-1])
                setattr(current_level, attrs[-1], convert_string_based_on_type(before_value, value))

        print_arguments(configs=self.configs)

        # Initialize feature extractor
        self._audio_featurizer = AudioFeaturizer(
            feature_method=self.configs.preprocess_conf.feature_method,
            method_args=self.configs.preprocess_conf.get('method_args', {})
        )

        # Load label list
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.strip() for l in lines]

        # Auto-detect number of classes
        if self.configs.model_conf.model_args.get('num_class', None) is None:
            self.configs.model_conf.model_args.num_class = len(self.class_labels)

        # Build model
        self.predictor = build_model(input_size=self._audio_featurizer.feature_dim, configs=self.configs)
        self.predictor.to(self.device)

        # Load model parameters
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pth')
        assert os.path.exists(model_path), f"Model does not exist: {model_path}"

        if torch.cuda.is_available() and use_gpu:
            model_state_dict = torch.load(model_path, weights_only=False)
        else:
            model_state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        self.predictor.load_state_dict(model_state_dict)
        print(f"Successfully loaded model parameters from: {model_path}")
        self.predictor.eval()

        # Load normalization scaler
        self.scaler = joblib.load(self.configs.dataset_conf.dataset.scaler_path)

    def _load_audio(self, audio_data, sample_rate=16000):
        """Load and preprocess audio input

        :param audio_data: Audio input; can be a file path, BufferedReader, bytes, or numpy array
        :param sample_rate: Sample rate (used if numpy array is provided)
        :return: Preprocessed feature
        """
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'Unsupported input type: {type(audio_data)}')

        if audio_segment.sample_rate != self.configs.dataset_conf.dataset.sample_rate:
            audio_segment.resample(self.configs.dataset_conf.dataset.sample_rate)

        if self.configs.dataset_conf.dataset.use_dB_normalization:
            audio_segment.normalize(target_db=self.configs.dataset_conf.dataset.target_dB)

        assert audio_segment.duration >= self.configs.dataset_conf.dataset.min_duration, \
            f'Audio too short. Minimum duration: {self.configs.dataset_conf.dataset.min_duration}s, got {audio_segment.duration}s'

        feature = self._audio_featurizer(audio_segment.samples, sample_rate=audio_segment.sample_rate)
        feature = self.scaler.transform([feature]).squeeze().astype(np.float32)
        return feature

    def predict(self, audio_data, sample_rate=16000):
        """Predict a single audio input

        :param audio_data: Input data (path, BufferedReader, bytes, or numpy array)
        :param sample_rate: Sample rate if input is numpy
        :return: (predicted label, confidence score)
        """
        if self.use_ms_model is not None:
            labels, scores = self.predictor.predict(audio_data)
            return labels[0], scores[0]

        input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device).unsqueeze(0)

        output = self.predictor(input_data)
        result = torch.nn.functional.softmax(output, dim=-1)[0].cpu().numpy()

        lab = np.argmax(result)
        score = result[lab]

        return self.class_labels[lab], round(float(score), 5)

    def predict_batch(self, audios_data: List, sample_rate=16000):
        """Predict a batch of audio inputs

        :param audios_data: List of audio inputs
        :param sample_rate: Sample rate if input is numpy
        :return: (list of predicted labels, list of confidence scores)
        """
        if self.use_ms_model is not None:
            labels, scores = self.predictor.predict(audios_data)
            return labels, scores

        audios_data1 = [self._load_audio(audio_data=a, sample_rate=sample_rate) for a in audios_data]

        # Padding to max sequence length
        batch = sorted(audios_data1, key=lambda a: a.shape[0], reverse=True)
        max_audio_length = batch[0].shape[0]
        batch_size = len(batch)

        inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
        for i, tensor in enumerate(audios_data1):
            inputs[i, :tensor.shape[0]] = tensor

        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)

        output = self.predictor(inputs)
        results = torch.nn.functional.softmax(output, dim=-1).cpu().numpy()

        labels, scores = [], []
        for result in results:
            lab = np.argmax(result)
            score = result[lab]
            labels.append(self.class_labels[lab])
            scores.append(round(float(score), 5))

        return labels, scores
