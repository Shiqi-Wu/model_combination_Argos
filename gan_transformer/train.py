import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
import gan_transformer as transformer
from load_dataset import *

scaler_x = StandardScaler()
scaler_u = StandardScaler()

class Params:
    def __init__(self, n_features, n_inputs, h=8, d_model=128, d_ff=2048, 
                 dropout=0.1, attn_type = 'softmax', N=6):
        self.h = h
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.attn_type = attn_type
        self.N = N
        self.n_features = n_features
        self.n_inputs = n_inputs

def build_model(params):
    self_attn = transformer.MultiHeadedAttention(params)
    feed_forward = nn.Linear(params.d_model, params.d_model)
    encoder_layer = transformer.EncoderLayer(params, self_attn, feed_forward, params.dropout)
    encoder = transformer.Encoder(params, encoder_layer)
    decoder_layer = transformer.DecoderLayer(params, self_attn, self_attn, feed_forward, params.dropout)
    decoder = transformer.Decoder(params, decoder_layer)
    model = transformer.EncoderDecoder(params, nn.Linear(params.n_features + params.n_inputs, params.d_model), nn.Linear(params.n_inputs, params.d_model), encoder, decoder, nn.Linear(params.d_model, params.n_features))
    return model

def train_one_epoch(model, optim, loss_fn, train_loader, epoch):
    model.train()
    train_loss = 0
    
    print(f"Training Epoch: {epoch + 1}")
    for batch_idx, (x, y, u1, u2) in enumerate(train_loader, 1):
        optim.zero_grad()
        y_pred = model(x, u1, u2)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        
        if batch_idx % 100 == 0:  # 每100个batch输出一次信息
            print(f"Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader):.4f}")
    return train_loss / len(train_loader)

def test_one_epoch(model, loss_fn, test_loader, epoch):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for _, (x, y, u1, u2) in enumerate(test_loader, 1):
            y_pred = model(x, u1, u2)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

    
    print(f"Epoch {epoch + 1}, Testing Loss: {test_loss / len(test_loader):.4f}")
    return test_loss / len(test_loader)

def main():
    train_dataset, test_dataset, n_features, n_inputs = data_preparation_v1()
    params = Params(n_features, n_inputs)
    model = build_model(params)
    model = torch.load('model.pth')
    optim = Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    scheduler = StepLR(optim, step_size=100, gamma=0.8)
    n_epochs = 1000
    train_losses = list(np.load('train_losses.npy'))
    test_losses = list(np.load('test_losses.npy'))
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, optim, loss_fn, train_loader, epoch)
        test_loss = test_one_epoch(model, loss_fn, test_loader, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()
        torch.save(model, 'model.pth')
    np.save('train_losses.npy', train_losses)
    np.save('test_losses.npy', test_losses)
    return

def main_v2(predict_num = 10):
    train_dataset, test_dataset, n_features, n_inputs = data_preparation_v2(predict_num)
    params = Params(n_features, n_inputs)
    model = build_model(params)
    model = torch.load('model.pth')
    optim = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    scheduler = StepLR(optim, step_size=100, gamma=0.8)
    n_epochs = 3000
    train_losses = list(np.load('train_losses.npy'))
    test_losses = list(np.load('test_losses.npy'))
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, optim, loss_fn, train_loader, epoch)
        test_loss = test_one_epoch(model, loss_fn, test_loader, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()
        torch.save(model, 'model_v2_10.pth')
    np.save('train_losses_v2_10.npy', train_losses)
    np.save('test_losses_v2_10.npy', test_losses)
    return

if __name__ == "__main__":
   main_v2()