import glob
import random
from functools import lru_cache

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import librosa
import augly.audio as audaugs
import numpy as np

from utils.vad_classifier import VAD_Classifier
from config import config


SEED = config["seed"]
SAMPLE_RATE = config["sample_rate"]
FRAME_SIZE_MS = config["frame_size_ms"]
FRAME_SIZE = config["frame_size"]

SPEECH_DATA_PATHS = config["speech_data_paths"]
NOISE_DATA_PATHS = config["noise_data_paths"]
BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]


class SpeechFrameSet(Dataset):
    """
    Returns frame-wise features from dataset
    """
    def __init__(self, speech_files, noise_files, feature_extractor, vad: VAD_Classifier):
        self.speech_files = speech_files
        self.noise_files = noise_files

        self.feature_extractor = feature_extractor

        self.vad = vad

        self.num_frames = 0
        self.idx2file = list()
        for i, file in enumerate(speech_files):
            file_num_frames = int(librosa.get_duration(filename=file) * 1000 // FRAME_SIZE_MS)
            self.idx2file.extend(zip([i] * file_num_frames, range(file_num_frames)))
            self.num_frames += file_num_frames

        self.speech_transforms = audaugs.Compose([
            audaugs.Reverb(p=0.5),
            audaugs.ToMono()
        ])

        self.noise_transforms = audaugs.Compose([
            audaugs.Reverb(p=0.5),
            audaugs.Clicks(p=0.5),
            audaugs.ToMono()
        ])

    def __len__(self):
        return self.num_frames

    # shuffle=False on DataLoader to reduce computations and I/O operations
    @lru_cache(maxsize=1024)
    def load_speech_and_noise_audios(self, speech_file):
        noise_file = random.choice(self.noise_files)

        speech_audio, _ = librosa.load(speech_file, sr=SAMPLE_RATE, mono=True)
        noise_audio, _ = librosa.load(noise_file, sr=SAMPLE_RATE, mono=True)

        if len(speech_audio) < FRAME_SIZE or len(noise_audio) < FRAME_SIZE:
            raise ValueError("One of the speech/noise files is too small")

        aug_speech_audio, _ = self.speech_transforms(speech_audio, SAMPLE_RATE)
        noise_audio, _ = self.noise_transforms(noise_audio, SAMPLE_RATE)
        mix_audio, _ = audaugs.add_background_noise(
            audio=aug_speech_audio, sample_rate=SAMPLE_RATE,
            background_audio=noise_audio,
            snr_level_db=random.randint(-3, 5)
        )

        return speech_audio, noise_audio, mix_audio

    def __getitem__(self, idx):
        file_idx, frame_idx = self.idx2file[idx]

        speech_file = self.speech_files[file_idx]

        speech_audio, noise_audio, mix_audio = self.load_speech_and_noise_audios(speech_file)

        frame_start = FRAME_SIZE * frame_idx
        frame_end = frame_start + FRAME_SIZE

        result_audio = mix_audio[frame_start:frame_end]
        speech_audio = speech_audio[frame_start:frame_end]

        self.vad.reinit()
        is_speech = self.vad.is_speech(speech_audio)

        if random.random() < 0.5:
            # In half of the cases return non-speech audio
            is_speech = False
            if random.random() < 0.8:
                # Most of this non-speech audio should be noise
                frame_start = random.randint(0, len(noise_audio)-FRAME_SIZE-1)
                frame_end = frame_start + FRAME_SIZE
                result_audio = noise_audio[frame_start:frame_end]
            else:
                # And the rest - almost silence
                result_audio, _ = audaugs.change_volume(
                    audio=result_audio, sample_rate=SAMPLE_RATE,
                    volume_db=random.choice([-65, -70, -75, -80]),
                )

        feature_map = self.feature_extractor(result_audio)

        return torch.from_numpy(feature_map).float(), torch.as_tensor([int(is_speech)], dtype=torch.float)


class SpeechFileSet(SpeechFrameSet):
    """
    Returns file-wise features from dataset
    Optimized for validation and logging results for entire file
    """
    def __init__(self, speech_files, noise_files, feature_extractor, vad: VAD_Classifier):
        super(SpeechFileSet, self).__init__(speech_files, noise_files, feature_extractor, vad)

    def __len__(self):
        return len(self.speech_files)

    def __getitem__(self, idx):
        speech_file = self.speech_files[idx]

        speech_audio, noise_audio, mix_audio = self.load_speech_and_noise_audios(speech_file)
        result_audio = mix_audio

        if random.random() < 0.5:
            # In half of the cases return non-speech audio
            if random.random() < 0.8:
                # Most of this non-speech audio should be noise
                result_audio = noise_audio
            else:
                # And the rest - almost silence
                result_audio, _ = audaugs.change_volume(
                    audio=result_audio, sample_rate=SAMPLE_RATE,
                    volume_db=random.choice([-65, -70, -75, -80]),
                )
            labels = [False] * (len(result_audio) // FRAME_SIZE)
        else:
            labels = list()
            for i in range(0, len(speech_audio), FRAME_SIZE):
                frame = speech_audio[i:i + FRAME_SIZE]
                if len(frame) != FRAME_SIZE:
                    break
                self.vad.reinit()
                labels.append(self.vad.is_speech(frame))

        features = list()
        for i in range(0, len(result_audio), FRAME_SIZE):
            frame = result_audio[i:i + FRAME_SIZE]
            if len(frame) != FRAME_SIZE:
                break
            features.append(self.feature_extractor(frame))

        return torch.as_tensor(np.array(features)).float(), torch.as_tensor(labels, dtype=torch.float), torch.as_tensor(result_audio)


def get_dataloaders(feature_extractor, vad: VAD_Classifier):
    speech_files = list()
    for data_path in SPEECH_DATA_PATHS:
        speech_files.extend(glob.glob(data_path + '/**/*.wav', recursive=True))
        speech_files.extend(glob.glob(data_path + '/**/*.flac', recursive=True))

    noise_files = list()
    for data_path in NOISE_DATA_PATHS:
        noise_files.extend(glob.glob(data_path + '/**/*.wav', recursive=True))
        noise_files.extend(glob.glob(data_path + '/**/*.flac', recursive=True))

    train_speech_files, test_speech_files = train_test_split(speech_files, test_size=0.3, shuffle=True, random_state=SEED)
    test_speech_files, val_speech_files = train_test_split(test_speech_files, test_size=0.5, shuffle=True, random_state=SEED)

    train_dataset = SpeechFrameSet(train_speech_files, noise_files, feature_extractor, vad)
    val_dataset = SpeechFileSet(val_speech_files, noise_files, feature_extractor, vad)
    test_dataset = SpeechFileSet(test_speech_files, noise_files, feature_extractor, vad)

    # Keeping shuffle=False is important for caching and memoization
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return train_dataloader, val_dataloader, test_dataloader
