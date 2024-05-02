import os
import torch
from torch.utils.data import Dataset, DataLoader

class CustomEEGDataset(Dataset):
    def __init__(self, EEG_dir, transform=None, target_transform=None):
        self.raw = torch.load(EEG_dir)
        self.EEG = torch.stack([t[0] for t in self.raw])
        self.labels = torch.stack([t[1] for t in self.raw])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.EEG[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label