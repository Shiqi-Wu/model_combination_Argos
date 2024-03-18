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
import argparse
import os
import yaml
import matplotlib.pyplot as plt


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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Reaction-diffusion model')
    parser.add_argument('--config', type=str, help='configuration file path')
    args = parser.parse_args()
    return args

def read_config_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Process the inputs.')

#     # For string arguments, you don't need to explicitly set the type, 
#     parser.add_argument('--output_dir', required=False, default='../output', type=str, help='output dictionary')
#     parser.add_argument('--output_suffix', required=True, type=str, help='output_suffix')
    
#     # For boolean arguments, use the custom function to handle boolean values from strings.
#     parser.add_argument('--position_encode', required=False, default=True, type=str_to_bool, help='need position encoder or not')
    
#     parser.add_argument('--data', required=False, default='../data_March', type=str, help='data dictionary')
#     parser.add_argument('--model_name', required=False, default=None, type=str, help='preload model parameters')
    
#     # For numerical arguments, you can specify types like int or float.
#     parser.add_argument('--predict_num', required=False, default=10, type=int, help='predict number')
#     parser.add_argument('--window_size', required=False, default=140, type=int, help='window size')
#     parser.add_argument('--epoch',  required=False, default=2500, type=int, help='training epoch')
#     args = parser.parse_args()

#     return args

def str_to_bool(value):
    if isinstance(value, bool):
       return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def build_model(params):
    self_attn = transformer.MultiHeadedAttention(params)
    feed_forward = transformer.FeedForward(params)
    encoder_layer = transformer.EncoderLayer(params, self_attn, feed_forward, params.dropout)
    encoder = transformer.Encoder(params, encoder_layer)
    decoder_layer = transformer.DecoderLayer(params, self_attn, self_attn, feed_forward, params.dropout)
    decoder = transformer.Decoder(params, decoder_layer)
    model = transformer.EncoderDecoder(params, nn.Linear(params.n_features + params.n_inputs, params.d_model), nn.Linear(params.n_inputs, params.d_model), encoder, decoder, nn.Linear(params.d_model, params.n_features))
    return model

def build_model_position_emb(params):
    self_attn = transformer.MultiHeadedAttention(params)
    feed_forward = transformer.FeedForward(params)
    emb1 = transformer.Embedding_encoder(params)
    emb2 = transformer.Embedding_decoder(params)
    encoder_layer = transformer.EncoderLayer(params, self_attn, feed_forward, params.dropout)
    encoder = transformer.Encoder(params, encoder_layer)
    decoder_layer = transformer.DecoderLayer(params, self_attn, self_attn, feed_forward, params.dropout)
    decoder = transformer.Decoder(params, decoder_layer)
    model = transformer.EncoderDecoder(params, emb1, emb2, encoder, decoder, nn.Linear(params.d_model, params.n_features))
    return model

def build_model_pureFFW(params):
    params.N = 2 * params.N
    self_attn = transformer.MultiHeadedAttention(params)
    feed_forward = transformer.FeedForward(params)
    emb1 = nn.Linear(params.n_features + params.n_inputs, params.d_model)
    emb2 = nn.Linear(params.n_inputs, params.d_model)
    encoder_layer = transformer.EncoderLayer_pureFFW(params, feed_forward, params.dropout)
    encoder = transformer.Encoder(params, encoder_layer)
    decoder_layer = transformer.DecoderLayer_pureFFW(params, self_attn, feed_forward, params.dropout)
    decoder = transformer.Decoder(params, decoder_layer)
    model = transformer.EncoderDecoder(params, emb1, emb2, encoder, decoder, nn.Linear(params.d_model, params.n_features))
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
        
        if batch_idx % 100 == 0:
            print(f"Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4e}")
    
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader):.4e}")
    return train_loss / len(train_loader)

def test_one_epoch(model, loss_fn, test_loader, epoch):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for _, (x, y, u1, u2) in enumerate(test_loader, 1):
            y_pred = model(x, u1, u2)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

    
    print(f"Epoch {epoch + 1}, Testing Loss: {test_loss / len(test_loader):.4e}")
    return test_loss / len(test_loader)

# def main():
#     train_dataset, test_dataset, n_features, n_inputs = data_preparation_v1()
    
#     params = Params(n_features, n_inputs)
#     model = build_model(params)
#     # model = torch.load('model.pth')

#     optim = Adam(model.parameters(), lr=0.0001)
#     loss_fn = nn.MSELoss()

#     train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
#     scheduler = StepLR(optim, step_size=100, gamma=0.8)
    
#     n_epochs = 1000

#     train_losses = []
#     test_losses = []
#     # train_losses = list(np.load('train_losses.npy'))
#     # test_losses = list(np.load('test_losses.npy'))

#     for epoch in range(n_epochs):
#         train_loss = train_one_epoch(model, optim, loss_fn, train_loader, epoch)
#         test_loss = test_one_epoch(model, loss_fn, test_loader, epoch)
#         train_losses.append(train_loss)
#         test_losses.append(test_loss)
#         scheduler.step()
#         torch.save(model, 'model.pth')
#     np.save('train_losses.npy', train_losses)
#     np.save('test_losses.npy', test_losses)
#     return

def main_v2(config):
    train_dataset, test_dataset, n_features, n_inputs = data_preparation_v2(config)
    
    params = Params(n_features, n_inputs, h = config['h'], d_model = config['d_model'], d_ff = config['d_ff'], dropout = config['dropout'], attn_type = config['attn_type'], N = config['N_num'])
    if config['position_encode'] == True:
        model = build_model_position_emb(params)
    else:
        model = build_model(params)
    if config.get('model_name')!=None:
        model = torch.load(config['model_name'])


    optim = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    scheduler = StepLR(optim, step_size=100, gamma=0.8)
    n_epochs = config['epoch']

    train_losses = []
    test_losses = []

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'], exist_ok=True)

    model_output_path = os.path.join(config['output_dir'], f"model_{config['output_suffix']}.pth")
    train_loss_output_path = os.path.join(config['output_dir'], f"train_loss_{config['output_suffix']}.npy")
    test_loss_output_path = os.path.join(config['output_dir'], f"test_loss_{config['output_suffix']}.npy")
    train_test_loss_fig_output_path = os.path.join(config['output_dir'], f"train_test_loss_{config['output_suffix']}.png")

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, optim, loss_fn, train_loader, epoch)
        test_loss = test_one_epoch(model, loss_fn, test_loader, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()
        torch.save(model, model_output_path)

    np.save(train_loss_output_path, train_losses)
    np.save(test_loss_output_path, test_losses)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()

    plt.savefig(train_test_loss_fig_output_path)
    return

if __name__ == "__main__":
   args = parse_arguments()
   config = read_config_file(args.config)
   main_v2(config)