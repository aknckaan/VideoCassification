from torch.utils.data import Dataset
from pathlib import Path
import csv
from torchvision.io import read_video
import kornia.augmentation as K
import cv2
import torch
import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

def build_pipeline( video_dir, annot_path):
    video_label_map = []
    label_int = []
    i = 0
    with open(annot_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            i+=1
            if i ==1:
                continue

            if not row[1] in label_int:
                label_int += [row[1]]

            video_label_map += [[row[0], label_int.index(row[1])]]

    file_names = list(map(lambda x: "dataset/train/"+x[0], video_label_map))
    file_labels = list(map(lambda x: x[1], video_label_map))
    sequence_length = 8
    initial_prefetch_size = 16
    batch_size = 2

    @pipeline_def
    def GetMnistPipeline(device):
        video, labels = fn.readers.video(device="gpu", filenames=file_names, labels=file_labels, sequence_length=sequence_length,
                                        random_shuffle=True, initial_fill=initial_prefetch_size)
        
        
        return video, labels

    pipe = GetMnistPipeline("gpu",num_threads=1, batch_size=batch_size, device_id=0)
    return pipe
