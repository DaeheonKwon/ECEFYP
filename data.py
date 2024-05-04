import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

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
    
def get_dataloaders(dataset, batch_size=50):
    class_inds = [torch.where(dataset.labels == class_idx)[0]
                for class_idx in torch.unique(dataset.labels)]

    dataloaders = [
        DataLoader(
            dataset=Subset(dataset, inds),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False)
        for inds in class_inds]
    
    return dataloaders

def split_datasets(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, generator=None):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)