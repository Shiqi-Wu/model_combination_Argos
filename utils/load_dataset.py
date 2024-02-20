import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

scaler_x = StandardScaler()
scaler_u = StandardScaler()

def load_dataset(data_dict):
    time = []
    data = []
    I_p = []
    for _, contents in data_dict.items():
        time.append(contents['time'])
        data.append(contents['data'])
        I_p.append(contents['I_p'])
    
    data = np.array(data)
    x_data = data[:-1,:]
    y_data = data[1:,:]
    dt_data = (time[1] - time[0]) * np.ones((data.shape[0] -1 , 1))
    # I_p = np.reshape(np.array(I_p)[:-1], (-1,1))
    u1_data = np.concatenate((dt_data, np.reshape(np.array(I_p)[:-1], (-1,1)), np.reshape(np.array(I_p)[1:], (-1,1))), axis = 1)
    u2_data = np.concatenate((dt_data, np.reshape(np.array(I_p)[1:], (-1,1)), np.reshape(np.concatenate((np.array(I_p)[2:], np.array([I_p[0]]))), (-1,1))), axis = 1)
    return x_data, y_data, u1_data, u2_data

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

def data_preparation_v1():
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

    x_data_slices = []
    y_data_slices = []
    u1_data_slices = []
    u2_data_slices = []

    for i in range(0, x_data_scaled.shape[0], window_size):
        for j in range(window_size - predict_num + 1):
            x_slice = x_data[i+j:i+j+predict_num,:].reshape((1, predict_num, -1))
            y_slice = y_data[i+j:i+j+predict_num,:].reshape((1, predict_num, -1))
            u1_slice = u1_data[i+j:i+j+predict_num,:].reshape((1, predict_num, -1))
            u2_slice = u2_data[i+j:i+j+predict_num,:].reshape((1, predict_num, -1))

            x_data_slices.append(x_slice)
            y_data_slices.append(y_slice)
            u1_data_slices.append(u1_slice)
            u2_data_slices.append(u2_slice)

    x_data_scaled = np.concatenate(x_data_slices, axis = 0)
    y_data_scaled = np.concatenate(y_data_slices, axis = 0)
    u1_data_scaled = np.concatenate(u1_data_slices, axis = 0)
    u2_data_scaled = np.concatenate(u2_data_slices, axis = 0)

    shuffled_indices = np.arange(len(x_data_scaled))
    np.random.shuffle(shuffled_indices)

    x_data_scaled = x_data_scaled[shuffled_indices]
    y_data_scaled = y_data_scaled[shuffled_indices]
    u1_data_scaled = u1_data_scaled[shuffled_indices]
    u2_data_scaled = u2_data_scaled[shuffled_indices]

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
