import os
import shutil
import torch

import joblib
import numpy as np
from torch.utils.data import Dataset
from yeaudio.audio import AudioSegment
from yeaudio.augmentation import ReverbPerturbAugmentor
from yeaudio.augmentation import SpeedPerturbAugmentor, VolumePerturbAugmentor, NoisePerturbAugmentor

from mser.data_utils.featurizer import AudioFeaturizer


class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 audio_featurizer: AudioFeaturizer,
                 scaler_path=None,
                 max_duration=3,
                 min_duration=0.5,
                 mode='train',
                 sample_rate=16000,
                 aug_conf=None,
                 use_dB_normalization=True,
                 target_dB=-20):
        """Audio Dataset Loader

        Args:
            data_list_path: Path to the file containing audio paths and labels
            audio_featurizer: Audio feature extractor
            scaler_path: Path to normalization scaler file
            max_duration: Maximum allowed audio duration; longer samples will be cropped
            min_duration: Minimum audio duration; shorter samples will be filtered out
            aug_conf: Configuration for data augmentation
            mode: Dataset mode. In 'train' mode, augmentations may be applied
            sample_rate: Target sample rate
            use_dB_normalization: Whether to apply decibel normalization
            target_dB: Target dB for volume normalization
        """
        super(CustomDataset, self).__init__()
        assert mode in ['train', 'eval', 'create_data', 'extract_feature']
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self.speed_augment = None
        self.volume_augment = None
        self.noise_augment = None
        self.reverb_augment = None

        # Load data list
        with open(data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        # Setup augmentors if in train mode and augment config is provided
        if mode == 'train' and aug_conf is not None:
            self.get_augmentor(aug_conf)

        # Setup featurizer
        self.audio_featurizer = audio_featurizer

        # Load scaler for feature normalization if needed
        if scaler_path and self.mode != 'create_data':
            self.scaler = joblib.load(scaler_path)

    def __getitem__(self, idx):
        # Split data path and label
        data_path, label = self.lines[idx].replace('\n', '').split('\t')

        # If it's a precomputed .npy file, load it directly
        if data_path.endswith('.npy'):
            feature = np.load(data_path)
        else:
            # Load raw audio
            audio_segment = AudioSegment.from_file(data_path)

            # If too short, skip it (only in train mode)
            if self.mode == 'train':
                if audio_segment.duration < self.min_duration:
                    return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)

            # Resample if needed
            if audio_segment.sample_rate != self._target_sample_rate:
                audio_segment.resample(self._target_sample_rate)

            # Apply augmentation if in train mode
            if self.mode == 'train':
                audio_segment = self.augment_audio(audio_segment)

            # Apply decibel normalization
            if self._use_dB_normalization:
                audio_segment.normalize(target_db=self._target_dB)

            # Crop long audios (except in feature extraction mode)
            if self.mode != 'extract_feature' and audio_segment.duration > self.max_duration:
                audio_segment.crop(duration=self.max_duration, mode=self.mode)

            # Extract features
            feature = self.audio_featurizer(audio_segment.samples, sample_rate=audio_segment.sample_rate)

        # Normalize features
        if self.mode not in ['create_data', 'extract_feature']:
            feature = self.scaler.transform([feature])
            feature = feature.squeeze().astype(np.float32)

        return (np.array(feature, dtype=np.float32), np.array(int(label), dtype=np.int64), data_path)

    def __len__(self):
        return len(self.lines)

    # Setup data augmentors
    def get_augmentor(self, aug_conf):
        if aug_conf.speed is not None:
            self.speed_augment = SpeedPerturbAugmentor(**aug_conf.speed)
        if aug_conf.volume is not None:
            self.volume_augment = VolumePerturbAugmentor(**aug_conf.volume)
        if aug_conf.noise is not None:
            self.noise_augment = NoisePerturbAugmentor(**aug_conf.noise)
        if aug_conf.reverb is not None:
            self.reverb_augment = ReverbPerturbAugmentor(**aug_conf.reverb)

    # Apply augmentations
    def augment_audio(self, audio_segment):
        if self.speed_augment is not None:
            audio_segment = self.speed_augment(audio_segment)
        if self.volume_augment is not None:
            audio_segment = self.volume_augment(audio_segment)
        if self.noise_augment is not None:
            audio_segment = self.noise_augment(audio_segment)
        if self.reverb_augment is not None:
            audio_segment = self.reverb_augment(audio_segment)
        return audio_segment
