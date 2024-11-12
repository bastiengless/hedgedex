###############
# TODO :
# 
###############

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import logging
import os
from timm.models.layers import DropPath

#from bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(n_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)


    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # CAREFUL : I removed the sigmoid activation, the ouptut is now raw logits
        # Edit : BCEWithLogitsLoss includes a sigmoid function
        return self.outc(dec1)
    
    def evaluate(self, valloader, metrics, device):
        self.eval()  # Set model to evaluation mode
        scores = [0.0] * len(metrics)
        with torch.no_grad():
            for batch in valloader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = self.forward(images)

                for i, metric in enumerate(metrics):
                    scores[i] += metric(outputs, labels).item()
        for i, metric in enumerate(metrics):
            scores[i] /= len(valloader)
        return scores
    
    
class UNet_time(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_time, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def CBR3d(in_channels, out_channels):
            """Conv, BatchNorm, ReLU block"""
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def CBR_strided(in_channels, out_channels):
            """Strided convolution on the dime dimension to reduce it"""
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, stride=(2, 1, 1), kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )    
        
        

        self.enc1 = CBR3d(n_channels, 64)
        self.enc2 = CBR_strided(64, 128)
        self.enc3 = CBR3d(128, 256)
        self.enc4 = CBR_strided(256, 512)


        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.time_pool3 = nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        self.time_pool6 = nn.AvgPool3d(kernel_size=(6, 1, 1), stride=(6, 1, 1))
        self.time_pool12 = nn.AvgPool3d(kernel_size=(12, 1, 1), stride=(12, 1, 1))

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)


    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        pooled = self.time_pool3(self.pool(enc4)).squeeze(2)        
        bottleneck = self.bottleneck(pooled)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, self.time_pool3(enc4).squeeze(2)), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, self.time_pool6(enc3).squeeze(2)), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, self.time_pool6(enc2).squeeze(2)), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, self.time_pool12(enc1).squeeze(2)), dim=1)
        dec1 = self.dec1(dec1)

        # CAREFUL : I removed the sigmoid activation, the ouptut is now raw logits
        # Edit : BCEWithLogitsLoss includes a sigmoid function
        return self.outc(dec1)
    
    def evaluate(self, valloader, metrics, device):
        self.eval()  # Set model to evaluation mode
        scores = [0.0] * len(metrics)
        with torch.no_grad():
            for batch in valloader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = self.forward(images)

                for i, metric in enumerate(metrics):
                    scores[i] += metric(outputs, labels).item()
        for i, metric in enumerate(metrics):
            scores[i] /= len(valloader)
        return scores
    
    
    
 