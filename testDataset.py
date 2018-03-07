import torch
from torch.utils.data import Dataset
import glob
import os
from PIL import Image

class TestDataset(Dataset):
    "transform: a list of pytorch transformation"

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_files = glob.glob(os.path.join(root_dir, '*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        name = self.img_files[idx]
        img = Image.open(name)
        name = os.path.basename(name)
        if self.transform:
            imgs = []
            for trans in self.transform:
                imgs.append(trans(img))
                
            return (imgs, name)
        return (img, name)
        
