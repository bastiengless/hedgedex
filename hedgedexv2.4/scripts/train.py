"""
TODO :
- change the file namings to include a timestamp of the training termination
- check if accessing data works correctly
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, Dataset, DataLoader
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import random
import pandas as pd
import time
import datetime
import argparse
from model import *
from utils import TIFDataset, ToTensor
from utils import jaccard_index_prob, jaccard_loss, mcc, f1_score, overall_accuracy


def train_model(model, trainloader, valloader, criterion, optimizer, metrics, n_epochs, model_save_path, metrics_save_path, curve_save_path, device):
    # Initialize tracking variables
    training_losses = []
    validation_scores = [[] for _ in metrics]
    best_score = 0.0

    for batch in trainloader:
        n_bands = batch['image'].shape[1]
        break
    assert n_bands==model.n_channels, "Number of bands does not match model n_channels"

    # Start of the training loop
    for epoch in range(n_epochs):

        # Training step
        model.train()
        running_train_loss = 0.0

        for batch in trainloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(trainloader)
        training_losses.append(avg_train_loss)

        # Validation step
        # TODO : import eval_model
        scores = model.evaluate(valloader, metrics, device)
        for i, score in enumerate(scores):
            validation_scores[i].append(score)

        epoch_message = f'Epoch [{epoch+1}/{n_epochs}], Training Loss: {avg_train_loss:.4f}'
        for i, metric in enumerate(metrics):
            epoch_message += f', {metric.__name__}: {scores[i]:.4f}'
        # TODO : ca va aller dans un log txt pour l'instant. il me fait un log csv avec les metrics
        print(epoch_message)
        # Save training metrics to CSV
        df = pd.DataFrame({'Epoch': range(1, n_epochs+1), 'Loss': training_losses})
        additional_metrics_df = pd.DataFrame(validation_scores, columns=['Jaccard index', 'Matthews Correlation Coefficient', 'F1-Score', 'Accuracy'])
        df = pd.concat([df, additional_metrics_df], axis=1)
        df.to_csv(metrics_save_path, index=False)

        # Save epoch model if it outperforms the last best one
        if scores[1] > best_score:# and epoch > 130:
            best_score = scores[1]
            torch.save(model.state_dict(), model_save_path)
            print(f"Saving model ... ({model_save_path})")

    torch.save(model.state_dict(), 'last_terminal_model.pth')

    # Plot training curves
    plt.plot(training_losses, label='Training Loss')
    for i, metric in enumerate(metrics):
        plt.plot(validation_scores[i], label=metric.__name__)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(curve_save_path)
    
def load_data(data_dir, n_bands, bs):
    bands3 = ['B2', 'B3', 'B4']
    bands5 = ['B2', 'B3', 'B4', 'B8', 'ndvi']
    bands9 = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'ndvi']
    bands16 = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 'ndvi', 'elev', 'slope']
    if n_bands == 3:
        bands = bands3
    elif n_bands == 5:
        bands = bands5
    elif n_bands == 9:
        bands = bands9
    else:
        bands = bands16
    
    dataset = TIFDataset(root_dir=data_dir, transform=ToTensor(), selected_bands=bands)

    train_indices = random.sample(range(0, 1024), 896)
    val_indices = [i for i in range(0, 1024) if i not in train_indices]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    trainloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)
    valloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=2)
    print('Dataset loaded.')
    return trainloader, valloader
    
def main():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--timestamp', type=str, required=True, help='Timestamp for naming outputs')
    args = parser.parse_args()

    timestamp = args.timestamp
    start = time.time()
    
    
    # Hyperparameters
    bs = 4
    lr = 1e-4
    n_epochs = 10
    n_bands = 5
    loss = jaccard_loss
    model_save_path = f'../output/models/best_model_{timestamp}.pth'
    metrics_save_path = f'../output/metrics/training_metrics_{timestamp}.csv'
    curve_save_path = f'../output/plots/training_curve_{timestamp}.png'
    print(f"Training log for training at timestamp {timestamp}")
    print(f"batch size: {bs}\n learning rate: {lr}\n number of epochs: {n_epochs}\n number of bands: {n_bands}\n loss function: {loss.__name__}")
    
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Connect to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the data
    data_dir = '../../data/dataset04/'
    trainloader, valloader = load_data(data_dir, n_bands, bs)

    #############################
    # Initialize and train the model
    model = UNet(n_channels=n_bands, n_classes=1).to(device)

    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).to(device))
    criterion = loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    metrics = [jaccard_index_prob, mcc, f1_score, overall_accuracy]

    train_model(model, trainloader, valloader,
                criterion, optimizer, metrics=metrics, num_epochs=n_epochs,
                save_path=model_save_path, device=device)
    
    end = time.time()
    print(f"Training took {end-start} seconds.")
    