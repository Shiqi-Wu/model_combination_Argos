import torch
import numpy as np
import torch.nn as nn

class DicNN(nn.Module):
    """
    Trainable dictionaries
    """

    def __init__(self, n_input, layer_sizes=[64, 64], n_psi_train=64):
        super(DicNN, self).__init__()
        self.layer_sizes = layer_sizes

        # Input and hidden layers
        self.input_layer = nn.Linear(n_input, self.layer_sizes[0], bias=False)
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_sizes[i - 1], layer_sizes[i]) for i in range(1, len(layer_sizes))])
        self.hidden_layers.append(nn.Linear(layer_sizes[-1], n_psi_train))

        # Inverse layers
        self.inv_input_layer = nn.Linear(n_psi_train, self.layer_sizes[-1], bias=False)
        self.inv_hidden_layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i - 1]) for i in range(len(layer_sizes) - 1, 0, -1)])
        self.inv_hidden_layers.append(nn.Linear(self.layer_sizes[0], n_input))

        # Activation function
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return x

    def inv_forward(self, x):
        x = self.inv_input_layer(x)
        for layer in self.inv_hidden_layers:
            x = self.activation(layer(x))
        return x

class ModelPsi(nn.Module):
    def __init__(self, dic):
        super(ModelPsi, self).__init__()
        self.dic = dic

    def forward(self, x):
        return self.dic(x)

class KoopmanLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(KoopmanLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)

class ModelKoopman(nn.Module):
    def __init__(self, model_psi, k_layer):
        super(ModelKoopman, self).__init__()
        self.model_psi = model_psi
        self.k_layer = k_layer

    def forward(self, x, y):
        psi_x = self.model_psi(x)
        psi_y = self.model_psi(y)
        output_x = self.k_layer(psi_x)
        return output_x - psi_y

class ModelPredict(nn.Module):
    def __init__(self, model_psi, model_inv_psi, k_layer):
        super(ModelPredict, self).__init__()
        self.model_psi = model_psi
        self.model_inv_psi = model_inv_psi
        self.k_layer = k_layer

    def forward(self, x):
        psi_x = self.model_psi(x)
        psi_x_predict = self.k_layer(psi_x)
        return self.model_inv_psi(psi_x_predict)

class ModelAutoencoder(nn.Module):
    def __init__(self, model_psi, model_inv_psi):
        super(ModelAutoencoder, self).__init__()
        self.model_psi = model_psi
        self.model_inv_psi = model_inv_psi

    def forward(self, x):
        psi_x = self.model_psi(x)
        return self.model_inv_psi(psi_x)

def Build_model(n_input, layer_sizes, n_psi_train):
    
    # DicNN model
    dic = DicNN(n_input=n_input, layer_sizes = layer_sizes, n_psi_train=n_psi_train)

    # Model Psi
    model_psi = ModelPsi(dic)

    # Koopman Layer
    k_layer = KoopmanLayer(input_dim=n_psi_train, output_dim=n_psi_train)

    # Model Koopman
    model_koopman = ModelKoopman(model_psi, k_layer)

    # Model Inverse Psi
    model_inv_psi = ModelPsi(dic.inv_forward)

    # Model Predict
    model_predict = ModelPredict(model_psi, model_inv_psi, k_layer)

    # Model Autoencoder
    model_auto = ModelAutoencoder(model_psi, model_inv_psi)

    return model_psi, model_koopman, model_inv_psi, model_predict, model_auto


