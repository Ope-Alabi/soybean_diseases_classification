from dataloader import ImageDatasetLoader
from model import DiseaseClassifier
from plot import Plot_Model
import torch
import torch.nn as nn
import random
from datetime import timedelta


from socket import gethostname
import subprocess
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier

import os
import time
import sys

import logging

logging.basicConfig(level=logging.INFO)


# DIR_PATH = "C:/Users/alabi/OneDrive - University of North Carolina at Charlotte/Personal Projects/Machine Learning/ASDID/Datasets/Soybean_ML_orig_20"
DIR_PATH = "/Users/oalabi1/Desktop/PhD/Datasets/Soybean_ML_orig"
# DIR_PATH = "../Soybean_ML_orig"

def main(backbone, ms):
    # Hyperparameters
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.0001
    momentum = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDatasetLoader(DIR_PATH, batch_size=batch_size)
    logging.info("Got dataset")
    # sampler = DistributedSampler(dataset.image_datasets_T,num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    # dataset.update_dataloaders(sampler)
    logging.info("Set sampler")
    model = DiseaseClassifier(backbone=backbone, output_shape=len(dataset.class_names), device=device, load_pretrained=ms["pretrained"], freeze=ms["freeze"], pddd=ms["pddd"])
    model = model.to(device)
    logging.info("Initialized model")
    model.set_seeds()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # # model = DDP(model, device_ids=device_ids)
    logging.info("Start model training")
    # model.train_model(criterion, optimizer, dataset.dataloaders, num_epochs)
    # model.save_model()
    logging.info("Saved trained model")
    # model.test_model(dataset.dataloaders, dataset.class_names)
    print(model.summarize_model())
    
    # plot = Plot_Model(backbone=backbone, output_shape=len(dataset.class_names), device=device, pddd=ms["pddd"], load_pretrained=ms["pretrained"], freeze=ms["freeze"])
    # plot.plot()
    # model_summary = model.summarize_model()
    # print(model_summary)

    

if __name__ == '__main__':
    model_settings = {
        "pddd+freeze": {"pretrained": False, "freeze": True, "pddd": True},
        "pddd+unfreeze": {"pretrained": False, "freeze": False, "pddd": True},
        "imagenet-pretrained": {"pretrained": True, "freeze": False, "pddd": False},
        "scratch": {"pretrained": False, "freeze": False, "pddd": False}
    }

    backbones = ["resnet50", "efficientnet_b0", "densenet201"]

    # main(backbone=backbones[1], ms=model_settings["pddd+freeze"])
    main(backbone=backbones[1], ms=model_settings["imagenet-pretrained"])
# pddd+unfreeze
# imagenet-pretrained