import os

import soundcard
import soundfile


class RecordAudio:
    def __init__(self, channels=1, sample_rate=16000):
        # Recording parameters
        self.channels = channels
        self.sample_rate = sample_rate

        # Get the default microphone
        self.default_mic = soundcard.default_microphone()

    def record(self, record_seconds=3, save_path=None):
        """Record audio

        :param record_seconds: Recording duration in seconds, default is 3 seconds
        :param save_path: Path to save the recording (should end with .wav)
        :return: Recorded audio as a numpy array
        """
        print("Recording started...")
        num_frames = int(record_seconds * self.sample_rate)
        data = self.default_mic.record(samplerate=self.sample_rate, numframes=num_frames, channels=self.channels)
        audio_data = data.squeeze()
        print("Recording finished!")
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            soundfile.write(save_path, data=data, samplerate=self.sample_rate)
        return audio_data
