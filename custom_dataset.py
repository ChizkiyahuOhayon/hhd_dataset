from PIL import Image
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class HHD_dataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(label_csv)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)  # pd.read_csv() can give the length of csv file

    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]  # location should be in a list
        image_path = os.path.join(self.image_dir, image_name)
        image_array = Image.open(image_path)
        if self.transform:
            image_tensor = self.transform(image_array)
        label = torch.tensor(self.annotations.iloc[index, 1])
        return (image_tensor, label)  # in a tuple