import pytorch_lightning as pl

from models.naive_linear_net import NaiveLinearNet
from models.lit_classifier import LitClassifier
from data_processing.speechset import get_dataloaders
from config import config
from utils.initialization import seed_everything


SEED = config["seed"]
EPOCHS = config["epochs"]


if __name__ == "__main__":
    seed_everything(SEED)

    classifier = NaiveLinearNet()
    lit = LitClassifier(classifier)

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders()

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        val_check_interval=1.0,
        gpus=1,
        accelerator="gpu",
        devices=1
    )
    trainer.fit(
        model=lit,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
