# Imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import rasterio


# LOSSES and EVAL METRICS

def jaccard_index_prob(preds, targets, smooth=1e-6):
    """Take logit predictions and compute Jaccard index"""
    preds = torch.sigmoid(preds)  # Ensure preds are in range [0, 1]

    preds = preds.view(preds.size(0), -1)  # Flatten while keeping batch dimension
    targets = targets.view(targets.size(0), -1)  # Flatten while keeping batch dimension

    intersection = torch.sum(preds * targets, dim=1)
    union = torch.sum(preds, dim=1) + torch.sum(targets, dim=1) - intersection

    jaccard_score = (intersection + smooth) / (union + smooth)
    jaccard_score = torch.mean(jaccard_score)  # Average across the batch

    return jaccard_score

def jaccard_loss(preds, targets, smooth=1e-6):
    return 1 - jaccard_index_prob(preds, targets, smooth)


def dice_coef_prob(preds, targets, smooth=1e-6):
    """Take logit predictions and compute Dice coefficient"""
    preds = torch.sigmoid(preds)  # Ensure preds are in range [0, 1]

    preds = preds.view(preds.size(0), -1)  # Flatten while keeping batch dimension
    targets = targets.view(preds.size(0), -1)  # Flatten while keeping batch dimension

    intersection = torch.sum(preds * targets, dim=1)
    union = torch.sum(preds, dim=1) + torch.sum(targets, dim=1)

    dice_score = (2. * intersection + smooth) / (union + smooth)
    dice_score = torch.mean(dice_score)  # Average across the batch

    return dice_score

def dice_loss(preds, targets, smooth=1e-6):
    return 1 - dice_coef_prob(preds, targets, smooth)

def overall_accuracy(preds, targets):
    """Compute overall accuracy"""
    preds = torch.sigmoid(preds)  # Ensure preds are in range [0, 1]

    preds = (preds > 0.5).float()

    preds = preds.view(preds.size(0), -1)  # Flatten while keeping batch dimension
    targets = targets.view(preds.size(0), -1)  # Flatten while keeping batch dimension

    correct = torch.sum(preds == targets, dim=1)
    accuracy = torch.mean(correct.float()) / (256.0*256.0)  # Average across the batch

    return accuracy

def f1_score(preds, targets, smooth=1e-6):
    """Compute F1 score"""
    preds = torch.sigmoid(preds)  # Ensure preds are in range [0, 1]

    preds = (preds > 0.5).float()

    preds = preds.view(preds.size(0), -1)  # Flatten while keeping batch dimension
    targets = targets.view(preds.size(0), -1)  # Flatten while keeping batch dimension

    tp = torch.sum(preds * targets, dim=1)
    fp = torch.sum(preds * (1 - targets), dim=1)
    fn = torch.sum((1 - preds) * targets, dim=1)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    f1 = torch.mean(f1)  # Average across the batch

    return f1

def mcc(preds, targets):
    """Compute Matthews correlation coefficient"""
    preds = torch.sigmoid(preds)  # Ensure preds are in range [0, 1]

    preds = (preds > 0.5).float()

    preds = preds.view(preds.size(0), -1)  # Flatten while keeping batch dimension
    targets = targets.view(preds.size(0), -1)

    tp = torch.sum(preds * targets, dim=1)
    tn = torch.sum((1 - preds) * (1 - targets), dim=1)
    fp = torch.sum(preds * (1 - targets), dim=1)
    fn = torch.sum((1 - preds) * targets, dim=1)

    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    mcc = torch.mean(numerator / (denominator + 1e-6))  # Average across the batch

    return mcc


# DATASET STRUCTURE

class TIFDataset(Dataset):
    def __init__(self, root_dir, selected_bands, transform=None):
        self.root_dir = root_dir
        self.bands = selected_bands
        self.transform = transform
        self.file_names = [f for f in os.listdir(root_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_names[idx])
        with rasterio.open(img_path) as src:
            # Read the image data
            raster = src.read()

            # Read the image metadata
            band_names = src.descriptions
        
        # Make indices correspond to band_names
        indices = [band_names.index(band) for band in self.bands]
        indices.append(-1)

        # Only take selected bands
        raster = raster[indices, :, :]

        raster = raster.astype(np.float32)

        sample = {'raster': raster,
                  'name': self.file_names[idx]}
        
        if self.transform:
            sample = self.transform(sample)
            

        return sample

class ToTensor(object):
    def __call__(self, sample):
        
        raster, name = sample['raster'], sample['name']
        
        raster = torch.tensor(raster, dtype=torch.float32)
        transform = transforms.Compose([transforms.RandomCrop((64, 64))])
        
        raster = transform(raster)
        image = raster[:-1, :, :]
        label = raster[-1, :, :]
        label = torch.unsqueeze(label, 0)
                
        return {'image': image,
                'label': label,
                'name': name}
