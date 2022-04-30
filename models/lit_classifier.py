import torch
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional import f1_score, precision, recall
import wandb

from utils.visualization import plot_audio_with_vad


class LitClassifier(pl.LightningModule):
    def __init__(self, classifier, sample_rate):
        super().__init__()
        self.classifier = classifier
        self.sample_rate = sample_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.log('train_f1_thres=0.5', f1_score(y_hat, y.int(), threshold=0.5), on_step=True, on_epoch=False)
        self.log('train_precision_thres=0.5', precision(y_hat, y.int(), threshold=0.5), on_step=True, on_epoch=False)
        self.log('train_recall_thres=0.5', recall(y_hat, y.int(), threshold=0.5), on_step=True, on_epoch=False)
        # self.log('train_precision_thres=0.8', precision(y_hat, y.int(), threshold=0.8), on_step=True, on_epoch=False)
        # self.log('train_f1_thres=0.8', f1_score(y_hat, y.int(), threshold=0.8), on_step=True, on_epoch=False)
        # self.log('train_recall_thres=0.8', recall(y_hat, y.int(), threshold=0.8), on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, audio = batch
        y_hat = self.classifier(torch.squeeze(x, 0))
        y = y.T
        val_loss = F.binary_cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)
        self.log('val_f1', f1_score(y_hat, y.int()), on_step=False, on_epoch=True)
        self.log('val_precision', precision(y_hat, y.int()), on_step=False, on_epoch=True)
        self.log('val_recall', precision(y_hat, y.int()), on_step=False, on_epoch=True)

        if batch_idx < 5:
            wandb.log({
                f"Audio": wandb.Audio(
                    audio[0].detach().cpu().numpy(),
                    caption=str(batch_idx),
                    sample_rate=self.sample_rate
                ),
                f"Audio with VAD": plot_audio_with_vad(
                    audio[0].detach().cpu().numpy(),
                    y_hat.detach().cpu().numpy(),
                    y.detach().cpu().numpy(),
                    str(batch_idx)
                )
            })

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier(x)
        test_loss = F.binary_cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
