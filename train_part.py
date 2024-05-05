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
import logging

def train_epoch(model, dataloaders, optimizer, loss_type, device):
    model.train()
    start_epoch = time.perf_counter()
    npc_losses = 0.
    iCNN_losses = 0.
    iterators = list(map(iter, dataloaders))
    total_length = sum(len(itr) for itr in iterators)
    report_interval = total_length // 10
    itr = 0
    while iterators:
        iterator = np.random.choice(iterators)
        try:
            itr += 1
            images, labels = next(iterator)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            '''Key Part: npc loss only updates npc positions, and iCNN loss only updates iCNN parameters'''
            npc_loss, iCNN_loss = loss_type(model(images), model)
            npc_losses += npc_loss.item()
            iCNN_losses += iCNN_loss.item()
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.npc.position.requires_grad = True
            npc_loss.backward(retain_graph=True)
            for name, param in model.named_parameters():
                if name != 'npc.label':
                    param.requires_grad = True
            model.npc.position.requires_grad = False
            iCNN_loss.backward()
            optimizer.step()
            if itr % report_interval == 0:
                print(f'Processed {itr}/{total_length} samples')
                logging.info(f'Processed {itr}/{total_length} samples')
        except StopIteration:
            iterators.remove(iterator)
    return npc_losses/total_length, iCNN_losses/total_length, time.perf_counter() - start_epoch

'''Calibration: 2 minutes of seizure-free data / labeling NPC clusters'''
def calibrate(model, dataloader, device): 
    model.eval()
    start = time.perf_counter()
    iterator = iter(dataloader)
    with torch.no_grad():  # using context manager
        images, _ = next(iterator) # batched image of batch size 120 : 2-min readout of non-seizure data
        images = images.to(device)
        output = model(images)
        label_count = torch.zeros_like(model.npc.label)
        for img in output:
            distances = torch.norm(img.view(1, -1, 1) - model.npc.position.data, dim=1).squeeze()
            closest_position_index = torch.argmin(distances)
            label_count[closest_position_index] += 1

        model.npc.label = nn.Parameter(torch.where(label_count > 0, 0, 1), requires_grad=False)
        print('label count:', label_count)
        logging.info(f'label count: {label_count}')
        print('npc label:', model.npc.label)
        logging.info(f'npc label: {model.npc.label}')
        # configurable. 0 (non-seizure) if at least 2 samples are closest to that particular NPC. Otherwise, 1 (seizure)

    return time.perf_counter() - start

def validate(model, dataloader, loss_type, device):
    model.eval()
    start = time.perf_counter() 
    confusion_matrix = np.zeros((2, 2))
    event_confusion_matrix = np.zeros((2, 2))
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
                _, pred= loss_type(output, model)
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

        cluster_labels = np.array([])
        cluster_preds = np.array([])

        for i, cluster in enumerate(clusters):
            if np.count_nonzero(cluster) > 0:
                cluster_labels = np.append(cluster_labels, 1)
                if np.count_nonzero(pred_clusters[i]) > 0:
                    cluster_preds = np.append(cluster_preds, 1)
                else:
                    cluster_preds = np.append(cluster_preds, 0)
            else:
                cluster_labels = np.append(cluster_labels, cluster)
                cluster_preds = np.append(cluster_preds, pred_clusters[i])

        event_confusion_matrix[0, 0] = np.sum(np.logical_and(cluster_preds == 0, cluster_labels == 0))
        event_confusion_matrix[0, 1] = np.sum(np.logical_and(cluster_preds == 0, cluster_labels == 1))
        event_confusion_matrix[1, 0] = np.sum(np.logical_and(cluster_preds == 1, cluster_labels == 0))
        event_confusion_matrix[1, 1] = np.sum(np.logical_and(cluster_preds == 1, cluster_labels == 1))
        

    return confusion_matrix, event_confusion_matrix, calibrate_time, time.perf_counter() - start
