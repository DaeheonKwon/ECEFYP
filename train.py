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

def train_epoch(model, dataloaders, optimizer, loss_type, device):
    model.train()
    start_epoch = time.perf_counter()
    loss = 0.
    iterators = list(map(iter, dataloaders))
    total_length = sum([sum([len(i) for i in itr]) for itr in iterators])
    while iterators:
        iterator = np.random.choice(iterators)
        try:
            images, labels = next(iterator)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            train_loss = loss_type(model(images), labels, model)
            loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
        except StopIteration:
            iterators.remove(iterator)
    return loss/total_length, time.perf_counter() - start_epoch

'''Calibration: 2 minutes of seizure-free data / labeling NPC clusters'''
def calibrate(model, dataloader, device): 
    model.eval()
    start = time.perf_counter()
    iterator = iter(dataloader)
    with torch.no_grad():  # using context manager
        for _ in range(120): # 1 samples per batch, 120 batches: 2-minute calibration
            images, _ = next(iterator)
            images = images.to(device)
            output = model(images)
            mean_output = torch.mean(output, dim=0) # unnecessary, since batch size is 1. left for consistency
            distances = torch.norm(mean_output.view(1, -1, 1) - model.npc.position.data, dim=1).squeeze()
            closest_position_index = torch.argmin(distances)
            model.npc.label[closest_position_index] = 0
    return time.perf_counter() - start

def validate(model, dataloader, loss_type, device):
    model.eval()
    start = time.perf_counter()
    loss = 0.
    confusion_matrix = np.zeros((2, 2))
    event_confusion_matrix = np.zeros((2, 2))
    total_length = len(dataloader)
    pred_all = np.array([])
    labels_all = np.array([])
    calibrate_time = calibrate(model, dataloader, device)  
    with torch.no_grad():  # using context manager
        iterator = iter(dataloader)
        while True:
            try:
                images, labels = next(iterator)
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss = loss_type(output, model)
                loss += val_loss[0].item()
                pred = val_loss[1].item()
                pred_all = np.append(pred_all, pred)
                labels = labels.cpu().numpy()
                labels_all = np.append(labels_all, labels)
            except StopIteration:
                break

        confusion_matrix[0, 0] = np.sum(np.logical_and(pred_all == 0, labels_all == 0))
        confusion_matrix[0, 1] = np.sum(np.logical_and(pred_all == 0, labels_all == 1))
        confusion_matrix[1, 0] = np.sum(np.logical_and(pred_all == 1, labels_all == 0))
        confusion_matrix[1, 1] = np.sum(np.logical_and(pred_all == 1, labels_all == 1))
        
        '''Constructing event-based confusion matrix'''
        clusters = np.split(labels_all, np.where(np.diff(labels_all) != 0)[0]+1)
        pred_clusters = np.split(pred_all, np.where(np.diff(labels_all) != 0)[0]+1)

        cluster_labels = []
        cluster_preds = []

        for i, cluster in enumerate(clusters):
            if np.count_nonzero(cluster) > 0:
                cluster_labels.append(1)
                if np.count_nonzero(pred_clusters[i]) > 0:
                    cluster_preds.append(1)
                else:
                    cluster_preds.append(0)
            else:
                cluster_labels.append(cluster)
                cluster_preds.append(pred_clusters[i])

        event_confusion_matrix[0, 0] = np.sum(np.logical_and(cluster_preds == 0, cluster_labels == 0))
        event_confusion_matrix[0, 1] = np.sum(np.logical_and(cluster_preds == 0, cluster_labels == 1))
        event_confusion_matrix[1, 0] = np.sum(np.logical_and(cluster_preds == 1, cluster_labels == 0))
        event_confusion_matrix[1, 1] = np.sum(np.logical_and(cluster_preds == 1, cluster_labels == 1))
        

    return loss/total_length, confusion_matrix, event_confusion_matrix, calibrate_time, time.perf_counter() - start

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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

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
        scheduler.step()
        train_loss_list.append(train_loss)
        save_model('../model', epoch, model, optimizer)
        print(
            f'Trainloss = {train_loss:.4g} TrainTime = {train_time:.4f}s'
        )

    print('Training completed. Starting validation...')

    for validation_dataset in validation_datasets:
        _, val_dataloader = get_dataloaders(validation_dataset)
        val_dataloaders.append(val_dataloader)

    # validation dataloaders are separated by patient. individual calibration is needed
    sensitivity_list = []
    specificity_list = []

    event_sensitivity_list = []
    event_specificity_list = []

    '''Validation for each patient in the validation set.'''
    for i, val_dataloader in enumerate(val_dataloaders):
        print('Validation for patient #', i+1, '/', len(val_dataloaders))
        val_loss, confusion_matrix, event_confusion_matrix, calibrate_time, val_time = validate(model, val_dataloader, npc_validation_loss, device)
        sensitivity = confusion_matrix[1, 1]/(confusion_matrix[1, 1] + confusion_matrix[0, 1])
        specificity = confusion_matrix[0, 0]/(confusion_matrix[0, 0] + confusion_matrix[1, 0])
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

        event_sensitivity = event_confusion_matrix[1, 1]/(event_confusion_matrix[1, 1] + event_confusion_matrix[0, 1])
        event_specificity = event_confusion_matrix[0, 0]/(event_confusion_matrix[0, 0] + event_confusion_matrix[1, 0])
        event_specificity_list.append(event_specificity)
        event_sensitivity_list.append(event_sensitivity)

        print(
                f'ValLoss = {val_loss:.4g} ValTime = {val_time:.4f}s Calibration Time= {calibrate_time:.4f}s\n'
                f'Sample-based Sensitivity = {sensitivity:.4g} Sample-based Specificity = {specificity:.4g}\n'
                f'Event-based Sensitivity = {event_sensitivity:.4g} Event-based Specificity = {event_specificity:.4g}'
            )
    
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

    for i in range(3):
        print(f'---------------------Cross-Validation Fold # {i+1}---------------------')
        train(train_datasets=[datasets[idx] for idx in train_datasets[i]], validation_datasets=[datasets[idx] for idx in validation_datasets[i]], num_epochs=40)