"""
Author    Daeheon Kwon (2024)
Date      2024.05.04
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os
import logging
from itertools import chain
from matplotlib import pyplot as plt
from loss import npc_training_loss, npc_validation_loss
from model import SciCNN
from data import get_dataloaders, CustomEEGDataset, split_datasets
from train_part import train_epoch, validate

def save_model(exp_dir, fold, epoch, model, optimizer):
    torch.save(
        {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'exp_dir': exp_dir
        },
        f= exp_dir + f'/model_{epoch}_{fold}.pt'
    )

def train(fold_num, train_datasets, validation_datasets, num_epochs=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SciCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    train_dataloaders = []
    val_dataloaders = []

    for train_dataset in train_datasets:
        train_dataloader, _ = get_dataloaders(train_dataset)
        train_dataloaders.append(train_dataloader)

    train_dataloaders = list(chain(*train_dataloaders))
    npc_loss_list = []
    iCNN_loss_list = []

    for validation_dataset in validation_datasets:
        _, val_dataloader = get_dataloaders(validation_dataset)
        val_dataloaders.append(val_dataloader)

    for epoch in range(num_epochs):
        print(f'Epoch #{epoch+1}')
        logging.info(f'Epoch #{epoch+1}')
        npc_loss, iCNN_loss, train_time = train_epoch(model, train_dataloaders, optimizer, npc_training_loss, device)
        scheduler.step()
        npc_loss_list.append(npc_loss)
        iCNN_loss_list.append(iCNN_loss)
        save_model('../model', fold_num, epoch, model, optimizer)
        print(
            f'NPCloss = {npc_loss:.4f} iCNNloss = {iCNN_loss:.4f} TrainTime = {train_time:.4f}s'
        )
        logging.info(
            f'NPCloss = {npc_loss:.4f} iCNNloss = {iCNN_loss:.4f} TrainTime = {train_time:.4f}s'
        )

        print('Training completed. Starting validation...')
        logging.info('Training completed. Starting validation...')

        # validation dataloaders are separated by patient. individual calibration is needed
        sensitivity_list = np.array([])
        specificity_list = np.array([])

        event_sensitivity_list = np.array([])
        event_specificity_list = np.array([])

        '''Validation for each patient in the validation set.'''
        for i, val_dataloader in enumerate(val_dataloaders):
            print('Validation for patient #', i+1, '/', len(val_dataloaders))
            logging.info(f'Validation for patient # {i+1}/{len(val_dataloaders)}')
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
            logging.info(
                    f'Calibration Time= {calibrate_time:.4f}s ValTime = {val_time:.4f}s\n'
                    f'Sample-based Sensitivity = {sensitivity:.4f} Sample-based Specificity = {specificity:.4f}\n'
                    f'Event-based Sensitivity = {event_sensitivity:.4f} Event-based Specificity = {event_specificity:.4f}'
                )
            
        print(f'Validation for epoch #{epoch+1} completed. Saving results...')
        logging.info(f'Validation for epoch #{epoch+1} completed. Saving results...')

        print(f'sample-based sensitivity: {sensitivity_list.mean()}, sample-based specificity: {specificity_list.mean()}')
        logging.info(f'sample-based sensitivity: {sensitivity_list.mean()}, sample-based specificity: {specificity_list.mean()}')

        print(f'event-based sensitivity: {event_sensitivity_list.mean()}, event-based specificity: {event_specificity_list.mean()}')
        logging.info(f'event-based sensitivity: {event_sensitivity_list.mean()}, event-based specificity: {event_specificity_list.mean()}')

        np.save(f'../results/sensitivity_list_fold_{fold_num+1}_epoch_{epoch}.npy', sensitivity_list)
        np.save(f'../results/specificity_list_fold_{fold_num+1}_epoch_{epoch}.npy', specificity_list)
        np.save(f'../results/event_sensitivity_list_fold_{fold_num+1}_epoch_{epoch}.npy', event_sensitivity_list)
        np.save(f'../results/event_specificity_list_fold_{fold_num+1}_epoch_{epoch}.npy', event_specificity_list)

    y1 = np.array([])
    
    for i in train_loss_list:
        y1 = np.append(y1, i)
    
    x = np.arange(0, num_epochs)
    
    plt.plot(x, y1, 'r-.', label = 'train')
    plt.legend()
    
    plt.title('Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.savefig(f'../figures/loss_fold_{fold_num}.png', dpi=300)


if __name__ == '__main__':
    log_file = '/home/dhkwon/project/ECEFYP/train.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    path = '/home/dhyun/project/FYP'
    datasets = [
        # CustomEEGDataset(path + '/chb01.pt'),
        # CustomEEGDataset(path + '/chb02.pt'),
        # CustomEEGDataset(path + '/chb03.pt'),
        # CustomEEGDataset(path + '/chb04-1.pt'),
        # CustomEEGDataset(path + '/chb04-2.pt'),
        # CustomEEGDataset(path + '/chb05.pt'),
        # CustomEEGDataset(path + '/chb06.pt'),
        # CustomEEGDataset(path + '/chb07.pt'),
        # CustomEEGDataset(path + '/chb08.pt'),
        # CustomEEGDataset(path + '/chb09.pt'),
        # CustomEEGDataset(path + '/chb10.pt'),
        # CustomEEGDataset(path + '/chb11.pt'),
        # CustomEEGDataset(path + '/chb12.pt'),
        # CustomEEGDataset(path + '/chb13.pt'),
        # CustomEEGDataset(path + '/chb14.pt'),
        # CustomEEGDataset(path + '/chb15.pt'),
        # CustomEEGDataset(path + '/chb16.pt'),
        # CustomEEGDataset(path + '/chb17.pt'),
        # CustomEEGDataset(path + '/chb18.pt'),
        # CustomEEGDataset(path + '/chb19.pt'),
        # CustomEEGDataset(path + '/chb20.pt'),
        # CustomEEGDataset(path + '/chb21.pt'),
        # CustomEEGDataset(path + '/chb22.pt'),
        CustomEEGDataset(path + '/chb23.pt'),
        CustomEEGDataset(path + '/chb24.pt'),
    ]

    validation_datasets = [
        [0],
        [12, 18, 20],
        [0, 11, 13],
        [16, 21, 24],
        [2, 14, 15],
        [7, 9, 19],
        [1, 8, 22],
        [3, 4, 5, 17],
        [6, 10, 23]
    ]

    train_datasets = [
        [1],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 21, 22, 23, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 22, 23],
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24],
        [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24],
        [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24]
    ]    

    for i in range(1):
        print(f'---------------------Cross-Validation Fold # {i+1}---------------------')
        train(fold_num=i, train_datasets=[datasets[idx] for idx in train_datasets[i]], validation_datasets=[datasets[idx] for idx in validation_datasets[i]], num_epochs=100)
