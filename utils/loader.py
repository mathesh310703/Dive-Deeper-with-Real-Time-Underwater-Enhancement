import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd

class CustomLoader(Dataset):
    def __init__(self, dataset_root, transform=True):
        self.dataset_root = dataset_root
        self.transform = transform

        # Load CSV file
        csv_file = os.path.join(dataset_root, "dataset.csv")
        self.data_info = pd.read_csv(csv_file)

        self.distorted_list = self.data_info['distorted'].tolist()
        self.restored_list = self.data_info['restored'].tolist()

        # Image transforms
        self.transform_pipeline = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.distorted_list)

    def __getitem__(self, idx):
        distorted_path = os.path.join(self.dataset_root, self.distorted_list[idx])
        restored_path = os.path.join(self.dataset_root, self.restored_list[idx])

        distorted_img = Image.open(distorted_path).convert("RGB")
        restored_img = Image.open(restored_path).convert("RGB")

        if self.transform:
            distorted_img = self.transform_pipeline(distorted_img)
            restored_img = self.transform_pipeline(restored_img)

        return {
            'distorted': distorted_img,
            'restored': restored_img
        }
