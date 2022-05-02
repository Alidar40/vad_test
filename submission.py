import os
import glob
from pathlib import Path

import torch
import numpy as np
import wandb
import librosa
from tqdm import tqdm

from config import config
from models.model_definition import get_classifier_by_name
from models.lit_classifier import LitClassifier, LitClassifierEnsemble
from utils.visualization import plot_audio_with_vad

if __name__ == "__main__":
    SAMPLE_RATE = config["sample_rate"]
    FRAME_SIZE = config["frame_size"]
    MODEL = config["model"]
    WANDB_ARGS = config["wandb"]
    CKPT_PATH = config["ckpt_path"]
    SUBMISSION_DATA_PATHS = config["submission_data_paths"]
    THRESHOLDS = config["submission_thresholds"]

    classifier, feature_extractor = get_classifier_by_name(MODEL)

    if type(CKPT_PATH) is list:
        model = LitClassifierEnsemble(CKPT_PATH, SAMPLE_RATE)
    else:
        model = LitClassifier.load_from_checkpoint(CKPT_PATH)
    model.eval()

    wandb.init(project=WANDB_ARGS["project"], name=WANDB_ARGS["submission_name"], mode=WANDB_ARGS["mode"])

    files = list()
    for data_path in SUBMISSION_DATA_PATHS:
        files.extend(glob.glob(data_path + '/**/*.wav', recursive=True))

    predictions = dict()
    for idx, file in enumerate(tqdm(sorted(files), desc="Predicting")):
        audio, _ = librosa.load(file, sr=SAMPLE_RATE, mono=True)

        if len(audio) < FRAME_SIZE:
            raise ValueError("One of the speech/noise files is too small")

        features = list()
        for i in range(0, len(audio), FRAME_SIZE):
            frame = audio[i:i + FRAME_SIZE]
            if len(frame) != FRAME_SIZE:
                break
            features.append(feature_extractor(frame))

        with torch.no_grad():
            pred = model(torch.as_tensor(np.array(features)).float()).flatten().detach().cpu()

        predictions[Path(file).name] = pred

        if idx % 1 == 0:
            wandb.log({
                f"Audio": wandb.Audio(audio, sample_rate=SAMPLE_RATE),
                f"Audio with VAD": plot_audio_with_vad(audio, pred)
            })

    if not Path("submissions/").exists():
        os.makedirs("submissions/")
    for threshold in THRESHOLDS:
        with open(f"submissions/submission_{MODEL}_{threshold}.csv", "w") as f:
            for filename, preds in predictions.items():
                f.write(filename + "," + ",".join(str(int(p > threshold)) for p in preds) + '\n')
