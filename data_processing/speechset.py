import glob
import random
from functools import lru_cache

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import webrtcvad
import librosa
import augly.audio as audaugs

from config import config


SEED = config["seed"]
SAMPLE_RATE = config["sample_rate"]
FRAME_SIZE_MS = config["frame_size_ms"]
FRAME_SIZE = config["frame_size"]

SPEECH_DATA_PATHS = config["speech_data_paths"]
NOISE_DATA_PATHS = config["noise_data_paths"]
BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]


class SpeechSet(Dataset):
    def __init__(self, speech_files, noise_files, feature_extractor=None):
        self.speech_files = speech_files
        self.noise_files = noise_files
        self.feature_extractor = feature_extractor

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)

        self.num_frames = 0
        self.idx2file = list()
        for i, file in enumerate(speech_files):
            file_num_frames = int(librosa.get_duration(filename=file) * 1000 // FRAME_SIZE_MS)
            self.idx2file.extend(zip([i] * file_num_frames, range(file_num_frames)))
            self.num_frames += file_num_frames

        self.speech_transforms = audaugs.Compose([
            audaugs.Reverb(p=0.1),
            audaugs.ToMono()
        ])

        self.noise_transforms = audaugs.Compose([
            audaugs.Reverb(p=0.5),
            audaugs.Clicks(p=0.2),
            audaugs.ToMono()
        ])

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        file_idx, frame_idx = self.idx2file[idx]
        speech_file = self.speech_files[file_idx]
        frame_start = FRAME_SIZE * frame_idx
        frame_end = frame_start + FRAME_SIZE

        clean_audio, augmented_audio = self.get_audio(speech_file)
        augmented_audio = augmented_audio[frame_start:frame_end]
        clean_audio = clean_audio[frame_start:frame_end]

        is_speech = self.vad.is_speech(clean_audio.tobytes(), SAMPLE_RATE)

        if random.random() < 0.25:
            # Make almost silence
            augmented_audio, _ = audaugs.change_volume(
                audio=augmented_audio,
                sample_rate=SAMPLE_RATE,
                volume_db=-65,
            )
            is_speech = False

        return torch.from_numpy(augmented_audio), torch.tensor([int(is_speech)], dtype=torch.float)

    # shuffle=False
    @lru_cache(maxsize=2048)
    def get_audio(self, speech_file):
        noise_file = random.choice(self.noise_files)

        speech_audio, _ = librosa.load(speech_file, sr=SAMPLE_RATE, mono=True)
        speech_audio, _ = self.speech_transforms(speech_audio, SAMPLE_RATE)

        noise_audio, _ = librosa.load(noise_file, sr=SAMPLE_RATE, mono=True)
        noise_audio, _ = self.noise_transforms(noise_audio, SAMPLE_RATE)

        mix_audio, _ = audaugs.add_background_noise(
            audio=speech_audio,
            sample_rate=SAMPLE_RATE,
            background_audio=noise_audio,
            snr_level_db=random.randint(-3, 5)
        )

        return speech_audio, mix_audio


def get_dataloaders():
    speech_files = list()
    for data_path in SPEECH_DATA_PATHS:
        speech_files.extend(glob.glob(data_path + '/**/*.wav', recursive=True))
        speech_files.extend(glob.glob(data_path + '/**/*.flac', recursive=True))

    noise_files = list()
    for data_path in NOISE_DATA_PATHS:
        noise_files.extend(glob.glob(data_path + '/**/*.wav', recursive=True))
        noise_files.extend(glob.glob(data_path + '/**/*.flac', recursive=True))

    train_speech_files, test_speech_files = train_test_split(speech_files, test_size=0.1, random_state=SEED)
    test_speech_files, val_speech_files = train_test_split(test_speech_files, test_size=0.5, random_state=SEED)
    train_dataset = SpeechSet(train_speech_files, noise_files)
    val_dataset = SpeechSet(val_speech_files, noise_files)
    test_dataset = SpeechSet(test_speech_files, noise_files)

    # Keeping shuffle=False is important for caching and memoization
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_dataloader, val_dataloader, test_dataloader
