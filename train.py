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
    total_length = sum([len(itr) for itr in iterators])
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
def calibrate(model, iterator, device): 
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():  # using context manager
        for _ in range(240): # 64 samples per batch, 240 batches: 2-minute calibration
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

    with torch.no_grad():  # using context manager
        calibrate_time = calibrate(model, dataloader, device)  
        iterator = iter(dataloader)
        while True:
            try:
                images, labels = next(iterator)
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss = loss_type(output, model)
                loss += val_loss[0].item()
                pred = val_loss[1].item().numpy()
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



def save_model(exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'exp_dir': exp_dir
        },
        f= exp_dir + '/model.pt'
    )

    if is_new_best[0]:
        os.system(f'cp {exp_dir}/model.pt {exp_dir}/best_loss_model.pt')
    if is_new_best[1]:
        os.system(f'cp {exp_dir}/model.pt {exp_dir}/best_sensitivity_model.pt')
    if is_new_best[2]:
        os.system(f'cp {exp_dir}/model.pt {exp_dir}/best_specificity_model.pt')

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

    for validation_dataset in validation_datasets:
        _, val_dataloader = get_dataloaders(validation_dataset)
        val_dataloaders.append(val_dataloader)

    best_val_loss = 1.

    best_specificity = 0.
    best_sensitivity = 0.

    train_loss_list = []
    val_loss_list = []

    val_loss_log = np.empty((0, 2))
    for epoch in range(num_epochs):
        print(f'Epoch #{epoch}')

        train_loss, train_time = train_epoch(model, train_dataloaders, optimizer, npc_training_loss, device)
        val_loss, confusion_matrix, event_confusion_matrix, calibrate_time, val_time = validate(model, val_dataloader, npc_validation_loss, device)
        scheduler.step()

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = '../loss/val_loss_log'
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        sensitivity = confusion_matrix[1, 1]/(confusion_matrix[1, 1] + confusion_matrix[0, 1])
        specificity = confusion_matrix[0, 0]/(confusion_matrix[0, 0] + confusion_matrix[1, 0])

        event_sensitivity = event_confusion_matrix[1, 1]/(event_confusion_matrix[1, 1] + event_confusion_matrix[0, 1])
        event_specificity = event_confusion_matrix[0, 0]/(event_confusion_matrix[0, 0] + event_confusion_matrix[1, 0])

        is_new_best = [
            val_loss <= best_val_loss,
            sensitivity >= best_sensitivity,
            specificity >= best_specificity
        ]
        
        best_val_loss = min(val_loss, best_val_loss)
        best_specificity = max(specificity, best_specificity)
        best_sensitivity = max(sensitivity, best_sensitivity)

        save_model('../model', epoch, model, optimizer, best_val_loss, is_new_best)

        print(
            f'Trainloss = {train_loss:.4g} ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s Calibration = {calibrate_time:.4f}s\n'
            f'Sample-based Sensitivity = {sensitivity:.4g} Sample-based Specificity = {specificity:.4g}\n'
            f'Event-based Sensitivity = {event_sensitivity:.4g} Event-based Specificity = {event_specificity:.4g}\n'
        )

        if is_new_best[0]:
            print(f'New best val loss: {best_val_loss:.4g}')
        if is_new_best[1]:
            print(f'New best sensitivity: {best_sensitivity:.4g}')
        if is_new_best[2]:
            print(f'New best specificity: {best_specificity:.4g}')

    y1 = np.array([])
    y2 = np.array([])
    
    for i in train_loss_list:
        y1 = np.append(y1, i)
    
    for i in val_loss_list:
        y2 = np.append(y2, i)
    
    x = np.arange(0, num_epochs)
    
    plt.plot(x, y1, 'r-.', label = 'train')
    plt.plot(x, y2, 'b-.', label = 'val')
    plt.legend()
    
    plt.title('Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.savefig('../figures/loss.png', dpi=300)


if __name__ == '__main__':
    train_datasets = [
        [CustomEEGDataset('../data/chb21.pt'), CustomEEGDataset('../data/chb22.pt')],
        [CustomEEGDataset('../data/chb22.pt'), CustomEEGDataset('../data/chb23.pt')],
        [CustomEEGDataset('../data/chb23.pt'), CustomEEGDataset('../data/chb21.pt')],
    ]
    validation_datasets = [
        [CustomEEGDataset('../data/chb23.pt')],
        [CustomEEGDataset('../data/chb21.pt')],
        [CustomEEGDataset('../data/chb22.pt')],
    ]

    for i in range(3):
        print(f'---------------------Cross-Validation Fold # {i+1}---------------------')
        train(train_datasets=train_datasets[i], validation_datasets=validation_datasets[i], num_epochs=40)