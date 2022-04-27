from torch import optim, nn, utils
from torch.nn import functional as F
import pytorch_lightning as pl


class LitClassifier(pl.LightningModule):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier(x)
        val_loss = F.binary_cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier(x)
        test_loss = F.binary_cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
