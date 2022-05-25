from training_loop import VideoClassificationModel
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataloader import VideoLoader
import kornia.augmentation as K


transform = K.VideoSequential(
    K.Resize([224, 224]),
    K.RandomAffine(360),
    K.ColorJiggle(0.2, 0.3, 0.2, 0.3),
    K.RandomBoxBlur(kernel_size=(3, 3)),
    data_format="BCTHW",
    )

vcm = VideoClassificationModel(transforms=transform)
train_set = VideoLoader("dataset/train/", "dataset/train.csv")
test_set = VideoLoader("dataset/test/", "dataset/test.csv")
train_loader = DataLoader(train_set, batch_size=32)
val_loader = DataLoader(test_set, batch_size=32)
trainer = pl.Trainer(gpus=0, precision=16, limit_train_batches=0.5)
trainer.fit(vcm, train_loader, val_loader)