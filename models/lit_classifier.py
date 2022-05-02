import time

import numpy as np
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
        self.save_hyperparameters()
        self.classifier = classifier
        self.sample_rate = sample_rate

        self.predictions = None
        self.target = None
        self.avg_delay_ms = 0

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
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
        y_hat = self(torch.squeeze(x, 0))
        y = y.T
        val_loss = F.binary_cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)
        self.log('val_f1', f1_score(y_hat, y.int()), on_step=False, on_epoch=True)
        self.log('val_precision', precision(y_hat, y.int()), on_step=False, on_epoch=True)
        self.log('val_recall', precision(y_hat, y.int()), on_step=False, on_epoch=True)

        if batch_idx < 6:
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
        begin = time.time()
        y_hat = self(x)
        end = time.time()
        delay_ms = (end - begin) * 1000 / x.size()[0]
        return y_hat, y.int(), delay_ms

    def test_epoch_end(self, outputs):
        self.predictions = torch.cat([o[0] for o in outputs])
        self.target = torch.cat([o[1] for o in outputs])
        self.avg_delay_ms = np.mean([o[2] for o in outputs])

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, audio = batch
        y_hat = self(torch.squeeze(x, 0))

        if batch_idx % 100 == 0:
            wandb.log({
                f"Audio": wandb.Audio(
                    audio[0].detach().cpu().numpy(),
                    sample_rate=self.sample_rate
                ),
                f"Audio with VAD": plot_audio_with_vad(
                    audio[0].detach().cpu().numpy(),
                    y_hat.detach().cpu().numpy(),
                )
            })

        return y_hat

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class LitClassifierEnsemble(LitClassifier):
    def __init__(self, checkpoints, sample_rate):
        super(LitClassifierEnsemble, self).__init__(None, sample_rate)
        self.classifiers = [LitClassifier.load_from_checkpoint(ckpt) for ckpt in checkpoints]
        _ = [clf.eval() for clf in self.classifiers]

    def forward(self, x):
        return torch.mean(torch.stack([clf(x) for clf in self.classifiers]), dim=0)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_epoch_end(self, outputs):
        self.predictions = torch.cat([o[0] for o in outputs])
        self.target = torch.cat([o[1] for o in outputs])
        self.avg_delay_ms = np.mean([o[2] for o in outputs])
