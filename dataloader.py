from torch.utils.data import Dataset
from pathlib import Path
import csv
from torchvision.io import read_video
import kornia.augmentation as K

class VideoLoader(Dataset):
    def __init__(self, video_dir, annot_path):
        super().__init__()
        self.video_dir = video_dir
        self.video_label_map = []
        self.label_int = []
        i = 0
        with open(annot_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                i+=1
                if i ==1:
                    continue

                if not row[1] in self.label_int:
                    self.label_int += [row[1]]

                self.video_label_map += [[row[0], self.label_int.index(row[1])]]

    def __len__(self):
        return len(self.video_label_map)

    def __getitem__(self, idx):
        vid_name, label = self.video_label_map[idx]
        img_dir = Path(self.video_dir) / vid_name
        vid = read_video(str(img_dir))[0] / 255
        vid = K.Resize([224, 224])(vid)

        return vid, label

vl = VideoLoader("dataset/train", "dataset/train.csv")
vid, label = vl.__getitem__(0)
print(vid[1].shape)