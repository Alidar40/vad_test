import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.naive_linear_net import NaiveLinearNet
from models.lenet import LeNet8, LeNet32
from models.lit_classifier import LitClassifier
from data_processing.speechset import get_dataloaders
from data_processing.feature_extractors import nofeature_extractor, logfbank_8_extractor, logfbank_32_extractor
from config import config
from utils.initialization import seed_everything


SEED = config["seed"]
EPOCHS = config["epochs"]
LOG_EVERY_N_STEP = config["log_every_n_step"]
VAL_CHECK_INTERVAL = config["val_check_interval"]
MODEL = config["model"]
WANDB_ARGS = config["wandb"]

if __name__ == "__main__":
    seed_everything(SEED)

    if MODEL == "naive_linear":
        classifier = NaiveLinearNet()
        feature_extractor = nofeature_extractor
    elif MODEL == "lenet8":
        classifier = LeNet8()
        feature_extractor = logfbank_8_extractor
    elif MODEL == "lenet32":
        classifier = LeNet32()
        feature_extractor = logfbank_32_extractor
    else:
        raise ValueError("No model or bad model specified")

    lit = LitClassifier(classifier)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(feature_extractor)

    wandb_logger = WandbLogger(project=WANDB_ARGS["project"], name=WANDB_ARGS["name"], mode=WANDB_ARGS["mode"])

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        filename=MODEL+"-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        val_check_interval=VAL_CHECK_INTERVAL,
        logger=wandb_logger,
        log_every_n_steps=LOG_EVERY_N_STEP,
        callbacks=[checkpoint_callback],
        gpus=1,
        accelerator="gpu",
        devices=1
    )
    trainer.fit(
        model=lit,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
