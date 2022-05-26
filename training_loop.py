from network import VideoClassifier
import pytorch_lightning as pl
import torch
import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from dataloader import build_pipeline
import os
batch_size=2
sequence_length=8
initial_prefetch_size=16
video_directory = "dataset/train"
n_iter=6
# Path to MNIST dataset

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

    def setup(self, stage=None):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size
        pipe = build_pipeline("dataset/train", "dataset/train.csv")

        class LightningWrapper(DALIClassificationIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                # DDP is used so only one pipeline per process
                # also we need to transform dict returned by DALIClassificationIterator to iterable
                # and squeeze the lables
                out = out[0]
                return [out[k] if k != "label" else torch.squeeze(out[k]) for k in self.output_map]

        self.train_loader = LightningWrapper(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)

    def train_dataloader(self):
        return self.train_loader
