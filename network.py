import torch
from torch import nn
import numpy as np

class classification_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.LeakyReLU(1e-3))

    def forward(self, x):
        x = self.network(x)
        return x


class ClassificationHead(nn.Module):   
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.network = nn.Sequential(classification_layer(in_channels, in_channels//2), classification_layer(in_channels//2, in_channels//4), nn.Linear(in_channels//4, num_classes), nn.Softmax(1))
    def forward(self, x):
        x = self.network(x)
        return x


class Residual3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels//2, 1, 1, 0)
        self.conv2 = nn.Conv3d(in_channels//2, in_channels//2, 3, 1, 1)
        self.conv3 = nn.Conv3d(in_channels//2, in_channels, 1, 1, 0)
        self.act = nn.LeakyReLU(1e-3)


    def forward(self, x):

        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        x = self.conv3(x_) + x
        x = self.act(x)
        return x


class Extractor3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (np.array(kernel_size) - 1) // 2
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, 2, list(padding))
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.LeakyReLU(1e-3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Encoder3D(nn.Module):
    def __init__(self, reps):
        super().__init__()
        self.e_s = Extractor3D(3, 32, [7, 7, 7])
        e1 = Extractor3D(32, 64, [3, 3, 3])
        self.s1 = nn.Sequential(e1, *[Residual3D(64) for i in range(reps[0])])
        e2 = Extractor3D(64, 128, [3, 3, 3])
        self.s2 = nn.Sequential(e2, *[Residual3D(128) for i in range(reps[1])])
        e3 = Extractor3D(128, 256, [3, 3, 3])
        self.s3 = nn.Sequential(e3, *[Residual3D(256) for i in range(reps[2])])
        e4 = Extractor3D(256, 512, [3, 3, 3])
        self.s4 = nn.Sequential(e4, *[Residual3D(512) for i in range(reps[3])])
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.e_s(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.pool(x).flatten(1)
        return x


class VideoClassifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = Encoder3D([1, 1, 1, 1])
        self.classifier = ClassificationHead(512, classes)

    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = self.classifier(x)
        return x
