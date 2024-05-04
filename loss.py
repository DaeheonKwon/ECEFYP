"""
Author    Daeheon Kwon (2024)
Date      2024.05.04
"""

import torch

def npc_training_loss(output, label, model):
    # output: (batch_size, 16, 1)
    mean_output = torch.mean(output, dim=0)
    distances = torch.norm(mean_output.view(1, -1, 1) - model.npc.position.data, dim=1).squeeze()
    closest_position_index = torch.argmin(distances)
    closest_position = model.npc.position[closest_position_index]
    loss = torch.norm(mean_output - closest_position)
    return loss

def npc_validation_loss(output, model):
    mean_output = torch.mean(output, dim=0)
    
    distances = torch.norm(mean_output.view(1, -1, 1) - model.npc.position.data, dim=1).squeeze()
    
    # Sort distances in descending order and get the sorted indexes
    sorted_indexes = torch.argsort(distances, descending=False)
    # Sort model.npc.position using the sorted indexes
    sorted_labels = model.npc.label[sorted_indexes]
    sorted_positions = model.npc.position[sorted_indexes]

    for i, label in enumerate(sorted_labels):
        if label != 2:
            closest_position = sorted_positions[i]
            closest_label = label
            break

    return torch.norm(mean_output - closest_position), closest_label