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
    total_length = sum(len(itr) for itr in iterators)
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
    loss = 0.    confusion_matrix = np.zeros((2, 2))
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
        

    return confusion_matrix, event_confusion_matrix, calibrate_time, time.perf_counter() - start
