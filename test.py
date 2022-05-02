import os
from warnings import warn

import pytorch_lightning as pl
from torchmetrics.functional import roc, auroc
import numpy as np
import wandb

from data_processing.speechset import get_test_dataloader
from config import config
from utils.initialization import seed_everything
from models.model_definition import get_classifier_by_name, get_vad_by_name
from models.lit_classifier import LitClassifier, LitClassifierEnsemble
from utils.visualization import plot_roc_multiple


if __name__ == "__main__":
    SEED = config["seed"]
    SAMPLE_RATE = config["sample_rate"]
    EXTERNAL_VAD = config["external_vad"]
    TEST_MODELS = config["test_models"]
    WANDB_ARGS = config["wandb"]

    wandb.init(project=WANDB_ARGS["project"], name=WANDB_ARGS["test_name"], mode=WANDB_ARGS["mode"])

    trainer = pl.Trainer(gpus=0, accelerator="cpu", devices=1,)

    fp_rates = list()
    tp_rates = list()
    roc_aucs = list()
    model_names = list()

    columns = [
        "Model",
        "Average delay per frame on CPU (ms)",
        "FA = FR threshold",
        "Min FA value", "Min FA threshold",
        "Min FR value", "Min FR threshold",
    ]
    table = wandb.Table(columns=columns)

    for params in TEST_MODELS:
        MODEL = params["model"]
        CKPT_PATH = params["ckpt_path"]

        seed_everything(SEED)

        classifier, feature_extractor = get_classifier_by_name(MODEL)
        vad = get_vad_by_name(EXTERNAL_VAD)
        test_dataloader = get_test_dataloader(feature_extractor, vad)

        if type(CKPT_PATH) is list:
            model = LitClassifierEnsemble(CKPT_PATH, SAMPLE_RATE)
            trainer.test(model=model, dataloaders=test_dataloader)
        elif CKPT_PATH == "":
            warn(f"No checkpoint for {MODEL}")
            model = LitClassifier(classifier, SAMPLE_RATE)
            trainer.test(model=model, dataloaders=test_dataloader)
        elif not os.path.exists(CKPT_PATH):
            raise ValueError(f"Checkpoint path is invalid")
        else:
            model = LitClassifier(classifier, SAMPLE_RATE)
            trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=CKPT_PATH)

        FA = list()
        FR = list()
        FA_equal_FR_threshold = 1.0
        FA_min = 1.0
        FR_min = 1.0
        FA_min_value = 0
        FR_min_value = 0
        for threshold in range(1, 100):
            threshold /= 100
            pred = (model.predictions > threshold).int()
            FA.append(sum((model.target == 0) == (pred == 1)).item() / len(pred))
            FR.append(sum((model.target == 1) == (pred == 0)).item() / len(pred))
            if FA == FR:
                FA_equal_FR_threshold = min(threshold, FA_equal_FR_threshold)
                # FA_equal_FR_threshold = threshold

        FA_min_idx = np.argmin(FA)
        FA_min = round(FA[FA_min_idx], 2)
        FA_min_threshold = (1 + FA_min_idx) / 100

        FR_min_idx = np.argmin(FR)
        FR_min = round(FR[FR_min_idx], 2)
        FR_min_threshold = (1 + FR_min_idx) / 100

        fpr, tpr, thresholds = roc(model.predictions, model.target)
        roc_auc = auroc(model.predictions, model.target, pos_label=1)

        fp_rates.append(fpr)
        tp_rates.append(tpr)
        roc_aucs.append(roc_auc)
        model_names.append(MODEL)

        table.add_data(
            MODEL,
            model.avg_delay_ms,
            FA_equal_FR_threshold,
            FA_min,
            FA_min_threshold,
            FR_min,
            FR_min_threshold
        )

    wandb.log({"Statistics": table})
    wandb.log({f"ROC": plot_roc_multiple(fp_rates, tp_rates, roc_aucs, model_names)})
