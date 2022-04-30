import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from data_processing.speechset import get_train_val_dataloaders
from models.model_definition import get_classifier_by_name, get_vad_by_name
from models.lit_classifier import LitClassifier
from utils.initialization import seed_everything
from config import config


if __name__ == "__main__":
    SAMPLE_RATE = config["sample_rate"]
    SEED = config["seed"]
    MODEL = config["model"]
    EXTERNAL_VAD = config["external_vad"]
    CONTINUE_FROM_CKPT = config["continue_from_ckpt"]
    CKPT_PATH = config["ckpt_path"]
    EPOCHS = config["epochs"]
    LOG_EVERY_N_STEP = config["log_every_n_step"]
    VAL_CHECK_INTERVAL = config["val_check_interval"]
    WANDB_ARGS = config["wandb"]

    seed_everything(SEED)

    classifier, feature_extractor = get_classifier_by_name(MODEL)
    model = LitClassifier(classifier, SAMPLE_RATE)

    vad = get_vad_by_name(EXTERNAL_VAD)
    train_dataloader, val_dataloader = get_train_val_dataloaders(feature_extractor, vad)

    wandb_logger = WandbLogger(project=WANDB_ARGS["project"], name=WANDB_ARGS["name"], mode=WANDB_ARGS["mode"])

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        filename=MODEL+"-{epoch:02d}-{val_loss:.2f}",
        dirpath=f"./checkpoints/{wandb_logger.experiment.id[-8:]}/"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        val_check_interval=VAL_CHECK_INTERVAL,
        logger=wandb_logger,
        log_every_n_steps=LOG_EVERY_N_STEP,
        callbacks=[checkpoint_callback],
        gpus=1,
        accelerator="gpu",
        devices=1,
    )

    if CONTINUE_FROM_CKPT:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
    else:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

