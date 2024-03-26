from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../utils')
print(sys.path)
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import koopman_dir as km
from load_dataset import *
import argparse
import os
import yaml
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Reaction-diffusion model')
    parser.add_argument('--config', type=str, help='configuration file path')
    args = parser.parse_args()
    return args

def read_config_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def koopman_loss_function(model, x, y, u, nu, lambda_1 = 1, lambda_2 = 1, lambda_3 = 1):
    loss_fn = nn.MSELoss()

    # prediction loss
    y_pred = model(x, u, nu)
    loss_1 = loss_fn(y_pred, y)

    # autoencoder loss
    x_reconstructed = model.auto_state(x)
    loss_2 = loss_fn(x_reconstructed, x)

    # Koopman loss
    y_latent = model.encode(y)
    y_pred_latent = model.forward_latent(x, u, nu)
    loss_3 = loss_fn(y_latent, y_pred_latent)

    return lambda_1 * loss_1 + lambda_2 * loss_2 + lambda_3 * loss_3, loss_1, loss_2, loss_3

def train_one_epoch(model, optim, loss_fn, train_loader, epoch, device):
    train_loss, train_loss1, train_loss2, train_loss3 = 0, 0, 0, 0
    model.train()
    for i, data in enumerate(train_loader):
        x, y, u, nu = data
        x, y, u, nu = x.to(device), y.to(device), u.to(device), nu.to(device)
        optim.zero_grad()
        loss, loss_1, loss_2, loss_3 = loss_fn(model, x, y, u, nu)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        train_loss1 += loss_1.item()
        train_loss2 += loss_2.item()
        train_loss3 += loss_3.item()
    print(f'Epoch {epoch}, Loss: {train_loss}, Loss1: {train_loss1}, Loss2: {train_loss2}, Loss3: {train_loss3}')
    return train_loss, train_loss1, train_loss2, train_loss3


        

