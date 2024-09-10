"""
TODO :
- make hyperparameters passed as arguments
- try changing number of workers for data loading
- set default hyperparameters better
- put loss as a command line parameter
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
    print("Training started.")
    # Initialize tracking variables
    training_losses = []
    validation_scores = []
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
        scores = model.evaluate(valloader, metrics, device)
        validation_scores.append(scores)
        # for i, score in enumerate(scores):
        #     validation_scores[i].append(score)

        epoch_message = f'Epoch [{epoch+1}/{n_epochs}], Training Loss: {avg_train_loss:.4f}'
        for i, metric in enumerate(metrics):
            epoch_message += f', {metric.__name__}: {scores[i]:.4f}'
        print(epoch_message)

        # Save epoch model if it outperforms the last best one
        # if scores[1] > best_score:# and epoch > 130:
        #     best_score = scores[1]
        #     torch.save(model.state_dict(), model_save_path)
        #     print(f"Saving model ... ({model_save_path})")

    torch.save(model.state_dict(), model_save_path)
    
    # Save training metrics to CSV
    df = pd.DataFrame({'Epoch': range(1, n_epochs+1), 'Loss': training_losses})
    additional_metrics_df = pd.DataFrame(validation_scores, columns=['Jaccard index', 'MCC', 'F1-Score', 'Accuracy'])
    df = pd.concat([df, additional_metrics_df], axis=1)
    df.to_csv(metrics_save_path, index=False)

    # Plot training curves
    plt.plot(training_losses, label='Training Loss')
    for i, metric in enumerate(metrics):
        plt.plot(np.array(validation_scores)[:,i], label=metric.__name__)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(curve_save_path)
    
def load_data(data_dir, n_bands, bs, dataset_size):
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

    train_size = int(0.8*dataset_size)
    train_indices = random.sample(range(0, dataset_size), train_size)
    val_indices = [i for i in range(0, dataset_size) if i not in train_indices]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    trainloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    valloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0)
    print('Dataset loaded.')
    return trainloader, valloader
    

def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network model.")
    
    # Add arguments
    parser.add_argument('--timestamp', type=str, default='TESTTEST', required=True, help='Timestamp for naming outputs')
    parser.add_argument('--bs', type=int, default=4, help='Input batch size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train (default: 3)')
    parser.add_argument('--bands', type=int, default=5, help='Number of bands of data (default: 5)')
    parser.add_argument('--dssize', type=int, default=1024, help='Size of the dataset (default: 1024)')
    parser.add_argument('--data_dir', type=str, default='$HOME/scratch/data/dataset04/', help='Path to the data directory (default: $HOME/scratch/data/dataset04/)')
    parser.add_argument('--loss', type=str, default='jaccard_loss', help='Loss function to use (default: jaccard_loss)')
    parser.add_argument('--job_id', type=int, default=0, help='Job ID for the current run (default: 0)')

    # Parse arguments
    args = parser.parse_args()
    return args
    
def main():
    # Connect to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parse_args()

    timestamp = args.timestamp
    start = time.time()
    job_id = args.job_id
    
    
    # Hyperparameters
    bs = args.bs
    lr = args.lr
    n_epochs = args.epochs
    n_bands = args.bands
    dataset_size = args.dssize
    loss = jaccard_loss if args.loss == 'jaccard_loss' else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10])).to(device)
    hp_stamp = f'{job_id}_bs{bs}_lr{lr}_epochs{n_epochs}_bands{n_bands}_{args.loss}'
    model_save_path = f'hedgedexv2.5/output/models/model_{hp_stamp}.pth'
    metrics_save_path = f'hedgedexv2.5/output/metrics/metrics_{hp_stamp}.csv'
    curve_save_path = f'hedgedexv2.5/output/plots/curve_{hp_stamp}.png'
    data_dir = args.data_dir
    print(f"Training log for training at timestamp {timestamp}")
    print(f" batch size: {bs}\n learning rate: {lr}\n number of epochs: {n_epochs}\n number of bands: {n_bands}\n dataset size: {dataset_size}\n loss function: {args.loss}")
    
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load the data
    trainloader, valloader = load_data(data_dir, n_bands, bs, dataset_size)
    i = 0
    for batch in valloader:
        name = batch['name']
        print(name)
        i+=1
        if i == 5:
            break

    #############################
    # Initialize and train the model
    model = UNet(n_channels=n_bands, n_classes=1).to(device)
    print('Model initialized.')

    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).to(device))
    criterion = loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    metrics = [jaccard_index_prob, mcc, f1_score, overall_accuracy]

    train_model(model, trainloader, valloader,
                criterion, optimizer, metrics=metrics, n_epochs=n_epochs,
                model_save_path=model_save_path, metrics_save_path=metrics_save_path, curve_save_path=curve_save_path, device=device)
    
    end = time.time()
    
    training_time = end - start
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    print(f"Training time: {minutes}m{seconds}s.")
    
if __name__ == '__main__':
    main()
    
