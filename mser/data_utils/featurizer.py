import os
import shutil
import torch

import librosa
import numpy as np
from loguru import logger

class AudioFeaturizer(object):
    """
    A class for extracting audio features using either custom methods or a pretrained Emotion2Vec model.

    :param feature_method: Feature extraction method name ('CustomFeature' or 'Emotion2Vec')
    :type feature_method: str
    :param method_args: Optional dictionary of arguments for feature extraction
    :type method_args: dict
    """

    def __init__(self, feature_method='Emotion2Vec', method_args={}):
        super().__init__()
        self._method_args = method_args         # Arguments passed to the feature method
        self._feature_method = feature_method   # Feature extraction method to use
        self._feature_model = None              # Placeholder for Emotion2Vec model if used
        logger.info(f'Feature extraction method selected {self._feature_method}')  # Log selected feature method

    def __call__(self, x, sample_rate: float) -> np.ndarray:
        """
        Extract features from an audio signal.

        :param x: Audio waveform (1D numpy array)
        :param sample_rate: Sampling rate of the waveform
        :return: Feature array (2D numpy array: [T, F])
        """
        if self._feature_method == 'CustomFeature':
            return self.custom_features(x, sample_rate)
        elif self._feature_method == 'Emotion2Vec':
            return self.emotion2vec_features(x)
        else:
            raise Exception(f'Feature extraction method {self._feature_method} does not exist!')  # Raise error for unsupported method

    def emotion2vec_features(self, x) -> np.ndarray:
        """
        Extract features using a pretrained Emotion2Vec model.

        :param x: Audio waveform
        :return: Emotion2Vec features as numpy array
        """
        from mser.utils.emotion2vec_predict import Emotion2vecPredict

        # Load the model only once
        if self._feature_model is None:
            use_gpu = True if torch.cuda.is_available() else False
            self._feature_model = Emotion2vecPredict('iic/emotion2vec_base', revision="v2.0.4", use_gpu=use_gpu)

        # Extract features using Emotion2Vec model
        feats = self._feature_model.extract_features(x, self._method_args)
        return feats

    @staticmethod
    def custom_features(x, sample_rate: float) -> np.ndarray:
        """
        Extract handcrafted audio features using librosa.

        :param x: Audio waveform
        :param sample_rate: Sampling rate
        :return: Combined feature array of shape [T, F]
        """
        # Short-Time Fourier Transform magnitude
        stft = np.abs(librosa.stft(x))

        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T  # [T, 50]

        # Spectral centroid
        cent = librosa.feature.spectral_centroid(y=x, sr=sample_rate).T  # [T, 1]

        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=x).T  # [T, 1]

        # Chromagram
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate).T  # [T, 12]

        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=x, sr=sample_rate).T  # [T, 128]

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T  # [T, 7]

        # Zero-crossing rate
        zerocr = librosa.feature.zero_crossing_rate(x).T  # [T, 1]

        # RMS energy
        rms = librosa.feature.rms(S=stft).T  # [T, 1]

        # Ensure all features have the same time dimension
        min_len = min(
            mfcc.shape[0], cent.shape[0], flatness.shape[0], chroma.shape[0],
            mel.shape[0], contrast.shape[0], zerocr.shape[0], rms.shape[0]
        )

        # Truncate all features to the shortest length and concatenate along feature axis
        features = np.concatenate([
            mfcc[:min_len],
            cent[:min_len],
            flatness[:min_len],
            chroma[:min_len],
            mel[:min_len],
            contrast[:min_len],
            zerocr[:min_len],
            rms[:min_len]
        ], axis=1)  # Final shape: [T, F]

        return features.astype(np.float32)

    @property
    def feature_dim(self):
        """
        Return the feature dimensionality based on the method used.

        :return: Integer feature dimension
        """
        if self._feature_method == 'CustomFeature':
            return 312  # Sum of individual feature dimensions
        elif self._feature_method == 'Emotion2Vec':
            return 768  # Predefined Emotion2Vec embedding size
        else:
            raise Exception(f'Feature extraction method {self._feature_method}  does not exist!')
