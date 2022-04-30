from warnings import warn

import torch
import pytorch_lightning as pl

from data_processing.speechset import get_test_dataloader
from config import config
from utils.initialization import seed_everything
from models.model_definition import get_classifier_by_name, get_vad_by_name
from models.lit_classifier import LitClassifier


if __name__ == "__main__":
    SEED = config["seed"]
    SAMPLE_RATE = config["sample_rate"]
    EXTERNAL_VAD = config["external_vad"]
    TEST_MODELS = config["test_models"]

    trainer = pl.Trainer(
        gpus=0,
        accelerator="cpu",
        devices=1,
    )

    for params in TEST_MODELS:
        MODEL = params["model"]
        CKPT_PATH = params["ckpt_path"]

        seed_everything(SEED)

        classifier, feature_extractor = get_classifier_by_name(MODEL)
        model = LitClassifier(classifier, SAMPLE_RATE)

        vad = get_vad_by_name(EXTERNAL_VAD)
        test_dataloader = get_test_dataloader(feature_extractor, vad)

        if CKPT_PATH == "":
            warn(f"No checkpoint for {MODEL}")
            trainer.test(model=model, dataloaders=test_dataloader)
        else:
            trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=CKPT_PATH)
