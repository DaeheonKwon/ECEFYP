"""
Author    Daeheon Kwon (2024)
Date      2024.05.04
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os
from itertools import chain
from matplotlib import pyplot as plt
from loss import npc_training_loss, npc_validation_loss
from model import SciCNN
from data import get_dataloaders, CustomEEGDataset, split_datasets
from train_part import train_epoch, validate

def save_model(exp_dir, epoch, model, optimizer):
    torch.save(
        {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'exp_dir': exp_dir
        },
        f= exp_dir + '/model.pt'
    )

def train(train_datasets, validation_datasets, num_epochs=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SciCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_dataloaders = []
    val_dataloaders = []

    for train_dataset in train_datasets:
        train_dataloader, _ = get_dataloaders(train_dataset)
        train_dataloaders.append(train_dataloader)

    train_dataloaders = list(chain(*train_dataloaders))
    train_loss_list = []
    for epoch in range(num_epochs):
        print(f'Epoch #{epoch}')
        train_loss, train_time = train_epoch(model, train_dataloaders, optimizer, npc_training_loss, device)
        # scheduler.step()
        train_loss_list.append(train_loss)
        save_model('../model', epoch, model, optimizer)
        print(
            f'Trainloss = {train_loss:.4f} TrainTime = {train_time:.4f}s'
        )

    print('Training completed. Starting validation...')

    for validation_dataset in validation_datasets:
        _, val_dataloader = get_dataloaders(validation_dataset)
        val_dataloaders.append(val_dataloader)

    # validation dataloaders are separated by patient. individual calibration is needed
    sensitivity_list = np.array([])
    specificity_list = np.array([])

    event_sensitivity_list = np.array([])
    event_specificity_list = np.array([])

    '''Validation for each patient in the validation set.'''
    for i, val_dataloader in enumerate(val_dataloaders):
        print('Validation for patient #', i+1, '/', len(val_dataloaders))
        confusion_matrix, event_confusion_matrix, calibrate_time, val_time = validate(model, val_dataloader, npc_validation_loss, device)
        sensitivity = confusion_matrix[1, 1]/(confusion_matrix[1, 1] + confusion_matrix[0, 1])
        specificity = confusion_matrix[0, 0]/(confusion_matrix[0, 0] + confusion_matrix[1, 0])
        sensitivity_list = np.append(sensitivity_list, sensitivity)
        specificity_list = np.append(specificity_list, specificity)

        event_sensitivity = event_confusion_matrix[1, 1]/(event_confusion_matrix[1, 1] + event_confusion_matrix[0, 1])
        event_specificity = event_confusion_matrix[0, 0]/(event_confusion_matrix[0, 0] + event_confusion_matrix[1, 0])
        event_specificity_list = np.append(event_specificity_list, event_specificity)
        event_sensitivity_list = np.append(event_sensitivity_list, event_sensitivity)

        print(
                f'Calibration Time= {calibrate_time:.4f}s ValTime = {val_time:.4f}s\n'
                f'Sample-based Sensitivity = {sensitivity:.4f} Sample-based Specificity = {specificity:.4f}\n'
                f'Event-based Sensitivity = {event_sensitivity:.4f} Event-based Specificity = {event_specificity:.4f}'
            )

        print(f'Validation #{i+1}/{len(val_dataloaders)}completed. Saving results...')
        np.save(f'../results/sensitivity_list.npy', sensitivity_list)
        np.save(f'../results/specificity_list.npy', specificity_list)
        np.save(f'../results/event_sensitivity_list.npy', event_sensitivity_list)
        np.save(f'../results/event_specificity_list.npy', event_specificity_list)

    y1 = np.array([])
    
    for i in train_loss_list:
        y1 = np.append(y1, i)
    
    x = np.arange(0, num_epochs)
    
    plt.plot(x, y1, 'r-.', label = 'train')
    plt.legend()
    
    plt.title('Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.savefig('../figures/loss.png', dpi=300)


if __name__ == '__main__':

    datasets = [
        CustomEEGDataset('../data/chb20.pt'),
        CustomEEGDataset('../data/chb21.pt'),
        CustomEEGDataset('../data/chb23.pt')
    ]
    
    train_datasets = [
        [0, 1],
        [0, 2],
        [1, 2]
    ]

    validation_datasets = [
        [2],
        [1],
        [0]
    ]

    for i in range(1):
        print(f'---------------------Cross-Validation Fold # {i+1}---------------------')
        train(train_datasets=[datasets[idx] for idx in train_datasets[i]], validation_datasets=[datasets[idx] for idx in validation_datasets[i]], num_epochs=1)