import gan_transformer as transformer
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../utils')
print(sys.path)
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from load_dataset import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

def data_preparation():
    # Data preparation
    x_dataset = []
    y_dataset = []
    u1_dataset = []
    u2_dataset = []


    for suffix in range(10, 60):
        data_file_path = '../data/data_dict_' + str(suffix) + '.npy'
        
        # Check if the file exists before trying to load it
        if os.path.exists(data_file_path):
            data_dict = np.load(data_file_path, allow_pickle=True).item()
            x_data, y_data, u1_data, u2_data = load_dataset(data_dict)
            x_dataset.append(x_data)
            y_dataset.append(y_data)
            u1_dataset.append(u1_data)
            u2_dataset.append(u2_data)
        else:
            print(f"File not found: {data_file_path}")

    """
    x_data.shape = (n_samples, n_features)
    y_data.shape = (n_samples, n_features)
    u1_data.shape = (n_samples, n_inputs)
    u2_data.shape = (n_samples, n_inputs)
    """

    x_data = np.concatenate(x_dataset, axis = 0)
    y_data = np.concatenate(y_dataset, axis = 0)
    u1_data = np.concatenate(u1_dataset, axis = 0)
    u2_data = np.concatenate(u2_dataset, axis = 0)

    n_samples = x_data.shape[0]
    n_features = x_data.shape[1]
    n_inputs = u1_data.shape[1]

    # Split the data into training and testing sets

    scaler_x.fit(x_data)
    scaler_u.fit(u1_data)
    x_data_scaled = scaler_x.transform(x_data)
    y_data_scaled = scaler_x.transform(y_data)
    u1_data_scaled = scaler_u.transform(u1_data)
    u2_data_scaled = scaler_u.transform(u2_data)

    shuffled_indices = np.arange(len(x_data))
    np.random.shuffle(shuffled_indices)

    x_data_scaled = x_data_scaled[shuffled_indices]
    y_data_scaled = y_data_scaled[shuffled_indices]
    u1_data_scaled = u1_data_scaled[shuffled_indices]
    u2_data_scaled = u2_data_scaled[shuffled_indices]

    x_data_scaled = np.reshape(x_data_scaled, (n_samples, 1, n_features))
    y_data_scaled = np.reshape(y_data_scaled, (n_samples, 1, n_features))
    u1_data_scaled = np.reshape(u1_data_scaled, (n_samples, 1, n_inputs))
    u2_data_scaled = np.reshape(u2_data_scaled, (n_samples, 1, n_inputs))

    x_train, x_test = train_test_split(x_data_scaled, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(y_data_scaled, test_size=0.2, random_state=42)
    u1_train, u1_test = train_test_split(u1_data_scaled, test_size=0.2, random_state=42)
    u2_train, u2_test = train_test_split(u2_data_scaled, test_size=0.2, random_state=42)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    u1_train = torch.tensor(u1_train, dtype=torch.float32)
    u1_test = torch.tensor(u1_test, dtype=torch.float32)
    u2_train = torch.tensor(u2_train, dtype=torch.float32)
    u2_test = torch.tensor(u2_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train, u1_train, u2_train)
    test_dataset = TensorDataset(x_test, y_test, u1_test, u2_test)

    return train_dataset, test_dataset, n_features, n_inputs

def data_preparation_v2(predict_num):
    
    window_size = 150

    # Data preparation
    x_dataset = []
    y_dataset = []
    u1_dataset = []
    u2_dataset = []

    for suffix in range(10, 60):
        data_file_path = '../data/data_dict_' + str(suffix) + '.npy'
        
        # Check if the file exists before trying to load it
        if os.path.exists(data_file_path):
            data_dict = np.load(data_file_path, allow_pickle=True).item()
            x_data, y_data, u1_data, u2_data = load_dataset(data_dict)
            x_dataset.append(x_data[:window_size])
            y_dataset.append(y_data[:window_size])
            u1_dataset.append(u1_data[:window_size])
            u2_dataset.append(u2_data[:window_size])
        else:
            print(f"File not found: {data_file_path}")

    """
    x_data.shape = (n_samples, n_features)
    y_data.shape = (n_samples, n_features)
    u1_data.shape = (n_samples, n_inputs)
    u2_data.shape = (n_samples, n_inputs)
    """

    x_data = np.concatenate(x_dataset, axis = 0)
    y_data = np.concatenate(y_dataset, axis = 0)
    u1_data = np.concatenate(u1_dataset, axis = 0)
    u2_data = np.concatenate(u2_dataset, axis = 0)

    n_samples = x_data.shape[0]
    n_features = x_data.shape[1]
    n_inputs = u1_data.shape[1]

    # Split the data into training and testing sets

    scaler_x.fit(x_data)
    scaler_u.fit(u1_data)
    x_data_scaled = scaler_x.transform(x_data)
    y_data_scaled = scaler_x.transform(y_data)
    u1_data_scaled = scaler_u.transform(u1_data)
    u2_data_scaled = scaler_u.transform(u2_data)

    for i in range(0, x_data_scaled.shape[0], window_size):
        for j in range(window_size - predict_num + 1):
            x_slice = x_data[i+j:i+j+predict_num].reshape((1, 1, -1))
            y_slice = y_data[i+j:i+j+predict_num].reshape((1, 1, -1))
            u1_slice = u1_data[i+j:i+j+predict_num].reshape((1, 1, -1))
            u2_slice = u2_data[i+j:i+j+predict_num].reshape((1, 1, -1))

    x_data_slices.append(x_slice)
    y_data_slices.append(y_slice)
    u1_data_slices.append(u1_slice)
    u2_data_slices.append(u2_slice)


    shuffled_indices = np.arange(len(x_data))
    np.random.shuffle(shuffled_indices)

    x_data_scaled = x_data_scaled[shuffled_indices]
    y_data_scaled = y_data_scaled[shuffled_indices]
    u1_data_scaled = u1_data_scaled[shuffled_indices]
    u2_data_scaled = u2_data_scaled[shuffled_indices]

    x_data_scaled = np.reshape(x_data_scaled, (n_samples, 1, n_features))
    y_data_scaled = np.reshape(y_data_scaled, (n_samples, 1, n_features))
    u1_data_scaled = np.reshape(u1_data_scaled, (n_samples, 1, n_inputs))
    u2_data_scaled = np.reshape(u2_data_scaled, (n_samples, 1, n_inputs))

    x_train, x_test = train_test_split(x_data_scaled, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(y_data_scaled, test_size=0.2, random_state=42)
    u1_train, u1_test = train_test_split(u1_data_scaled, test_size=0.2, random_state=42)
    u2_train, u2_test = train_test_split(u2_data_scaled, test_size=0.2, random_state=42)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    u1_train = torch.tensor(u1_train, dtype=torch.float32)
    u1_test = torch.tensor(u1_test, dtype=torch.float32)
    u2_train = torch.tensor(u2_train, dtype=torch.float32)
    u2_test = torch.tensor(u2_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train, u1_train, u2_train)
    test_dataset = TensorDataset(x_test, y_test, u1_test, u2_test)

    return train_dataset, test_dataset, n_features, n_inputs

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
    train_dataset, test_dataset, n_features, n_inputs = data_preparation()
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

if __name__ == "__main__":
   main()