import torch
from torch import nn
from torchvision import models

import numpy as np
import os
import math
from PIL import Image
import matplotlib.pyplot as plt

# Helper classes and methods to plot the key metrics during a training run.

class Plot_Model(nn.Module):
    def __init__(self, backbone, pddd, load_pretrained, freeze, output_shape, device):
        super(Plot_Model, self).__init__()
        self.backbone = backbone
        self.load_pretrained = load_pretrained
        self.pddd = pddd
        self.freeze = freeze
        self.path = f"./OUTPUT/CHECKPOINT/{backbone}_best_model[pddd={self.pddd},frozen={self.freeze},pretrained={self.load_pretrained}].pth"

        if backbone == "densenet201":
            # Load the DENSENET201 model without pre-trained weights
            self.model = models.densenet201(weights=None).to(device)
            
            # Load the pre-trained weights
            state_dict = torch.load(self.path, map_location=device)
            
            # Modify the fully connected layer to match the state_dict
            num_ftrs = self.model.classifier.in_features  # Get the number of input features for the fc layer

            # Modify the classifier
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=num_ftrs, out_features=128, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=128, out_features=64, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(in_features=64, out_features=output_shape, bias=True)  # Number of output classes
            )

            # Load the pre-trained weights into the modified model
            self.model.load_state_dict(state_dict, strict=False)

        elif backbone == 'resnet50':
            # Load the ResNet50 model without pre-trained weights
            self.model = models.resnet50(weights=None).to(device)
            
            # Load the pre-trained weights
            state_dict = torch.load(self.path, map_location=device)
            
            # Modify the fully connected layer to match the state_dict
            num_ftrs = self.model.fc.in_features  # Get the number of input features for the fc layer

            # Modify the classifier
            self.model.fc = nn.Sequential(
                nn.Linear(in_features=num_ftrs, out_features=128, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=128, out_features=64, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(in_features=64, out_features=output_shape, bias=True)  # Number of output classes
            )

            # Load the pre-trained weights into the modified model
            self.model.load_state_dict(state_dict, strict=False)

        elif backbone == "vgg16":
            # Load the VGG16 model without pre-trained weights
            self.model = models.vgg16_bn(weights=None).to(device)
            
            # Load the pre-trained weights
            state_dict = torch.load(self.path, map_location=device)
            
            # Modify the fully connected layer to match the state_dict
            num_ftrs = self.model.classifier[6].in_features  # Get the number of input features for the fc layer

            # Modify the classifier
            self.model.classifier[6] = nn.Sequential(
                nn.Linear(in_features=num_ftrs, out_features=128, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=128, out_features=64, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(in_features=64, out_features=output_shape, bias=True)  # Number of output classes
            )

            # Load the pre-trained weights into the modified model
            self.model.load_state_dict(state_dict, strict=False)

        elif backbone == "efficientnet_b0":
            # Load the VGG16 model without pre-trained weights
            self.model = models.efficientnet_b0(weights=None).to(device)
            
            # Load the pre-trained weights
            state_dict = torch.load(self.path, map_location=device)
            
            # Modify the fully connected layer to match the state_dict
            num_ftrs = self.model.classifier[1].in_features  # Get the number of input features for the fc layer

            # Modify the classifier
            self.model.classifier[1] = nn.Sequential(
                nn.Linear(in_features=num_ftrs, out_features=128, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=128, out_features=64, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(in_features=64, out_features=output_shape, bias=True)  # Number of output classes
            )

            # Load the pre-trained weights into the modified model
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.model(x)
    
    def plot (self):
        history_dir = f"./OUTPUT/LOGS/{self.backbone}_train_hist[pddd={self.pddd},frozen={self.freeze},pretrained={self.load_pretrained}].txt"

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        total_sec = 0

        with open(history_dir, "r") as history:
            lines = history.readlines()

            # loss_acc = history.readlines()
            for line in lines[1:-1]:
                train_l, train_a, val_l, val_a, t = line.strip().split(", ")[1:]
                train_loss.append(float(train_l.split(": ")[1]))
                train_acc.append(float(train_a.split(": ")[1][:-1]))
                val_loss.append(float(val_l.split(": ")[1]))
                val_acc.append(float(val_a.split(": ")[1][:-1]))
                total_sec += (float(t.split(": ")[1].split(" ")[0]))
            
        epochs = len(lines) - 2

        print(f"Total Training Time: {(total_sec//60)//60} hr {(total_sec//60)%60} min {math.ceil(total_sec%60)} sec")


        # Create a figure with 1 row and 2 columns for side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot training & validation accuracy values
        axes[0].plot(train_acc, label='Train')
        axes[0].plot(val_acc, label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Number of Epochs')
        axes[0].legend(loc='upper left')

        # Plot training & validation loss values
        axes[1].plot(train_loss, label='Train')
        axes[1].plot(val_loss, label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Number of Epochs')
        axes[1].legend(loc='upper left')

        fig.suptitle(f"Total Training Time: {(total_sec//60)//60} hr. {(total_sec//60)%60} mins {math.ceil(total_sec%60)} sec")

        plt.tight_layout()

        fig.savefig(f"./OUTPUT/FIGURES/{self.backbone}_plot[pddd={self.pddd},frozen={self.freeze},pretrained={self.load_pretrained}].png")

        # plt.show()
