from network import VideoClassifier
import pytorch_lightning as pl
import torch


class VideoClassificationModel(pl.LightningModule):
    def __init__(self, transforms):
        super().__init__()
        self.network = VideoClassifier(10)
        self.transforms = transforms

    def forward(self, x):
        x = self.network(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=8e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = self.transforms(x)
        x = self.forward(x)
        loss = torch.nn.Functional.cross_entropy_loss(x, y)
        self.log("training_loss", loss)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = self.forward(x)
        loss = torch.nn.Functional.cross_entropy_loss(x, y)
        self.log('val_loss', loss)
