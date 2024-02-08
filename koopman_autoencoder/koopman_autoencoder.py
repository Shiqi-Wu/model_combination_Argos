import torch
import numpy as np
import torch.nn as nn

class DicNN(nn.Module):
    """
    Trainable dictionaries
    """

    def __init__(self, n_input, layer_sizes=[64, 64], n_psi_train=64, nonlinearity = True):
        super(DicNN, self).__init__()
        self.layer_sizes = layer_sizes
        self.nonlinearity = nonlinearity

        # Input and hidden layers
        self.input_layer = nn.Linear(n_input, self.layer_sizes[0], bias=False)
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_sizes[i - 1], layer_sizes[i]) for i in range(1, len(layer_sizes))])
        self.output_layer = nn.Linear(layer_sizes[-1], n_psi_train)

        # Inverse layers
        self.inv_input_layer = nn.Linear(n_psi_train, self.layer_sizes[-1], bias=False)
        self.inv_hidden_layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i - 1]) for i in range(len(layer_sizes) - 1, 0, -1)])
        self.inv_output_layer = nn.Linear(self.layer_sizes[0], n_input)

        # Activation function
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        if self.nonlinearity:
            for layer in self.hidden_layers:
                x = self.activation(layer(x))
        else:
            for layer in self.hidden_layers:
                x = layer(x)
        x = self.output_layer(x)
        return x

    def inv_forward(self, x):
        x = self.inv_input_layer(x)
        if self.nonlinearity:
            for layer in self.inv_hidden_layers:
                x = self.activation(layer(x))
        else:
            for layer in self.inv_hidden_layers:
                x = layer(x)
        x = self.inv_output_layer(x)
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
    
class ParaKoopmanLayer(nn.Module):
    def __init__(self, layer_sizes = [64, 64], input_dim = 2, K_dim = 64):
        super(ParaKoopmanLayer, self).__init__()

        self.K_dim = K_dim
        self.layer_sizes = layer_sizes

        # Input and hidden layers
        self.input_layer = nn.Linear(input_dim, self.layer_sizes[0], bias=False)
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_sizes[i - 1], layer_sizes[i]) for i in range(1, len(layer_sizes))])
        self.output_layer = nn.Linear(layer_sizes[-1], K_dim ** 2)

        # Activation function
        self.activation = nn.Tanh()

    def forward(self, u):
        K = self.input_layer(u)
        for layer in self.hidden_layers:
            K = self.activation(layer(K))
        K = self.output_layer(K)
        return torch.reshape(K, (-1, self.K_dim, self.K_dim))
    
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
    
class ModelParaKoopman(nn.Module):
    def __init__(self, model_psi, Para_k):
        super(ModelParaKoopman, self).__init__()
        self.model_psi = model_psi
        self.Para_k = Para_k

    def forward(self, x, y, u):
        psi_x = self.model_psi(x)
        psi_y = self.model_psi(y)

        psi_x_expanded = psi_x.unsqueeze(1)
        output_x = torch.bmm(psi_x_expanded, self.Para_k(u))
        output_x = output_x.squeeze(1)
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

class ModelPredictPara(nn.Module):
    def __init__(self, model_psi, model_inv_psi, Para_k):
        super(ModelPredictPara, self).__init__()
        self.model_psi = model_psi
        self.model_inv_psi = model_inv_psi
        self.Para_k = Para_k

    def forward(self, x, u):
        psi_x = self.model_psi(x)
        psi_x_expanded = psi_x.unsqueeze(1)
        psi_x_predict = torch.bmm(psi_x_expanded, self.Para_k(u))
        psi_x_predict = psi_x_predict.squeeze(1)
        return self.model_inv_psi(psi_x_predict)
    
class ModelAutoencoder(nn.Module):
    def __init__(self, model_psi, model_inv_psi):
        super(ModelAutoencoder, self).__init__()
        self.model_psi = model_psi
        self.model_inv_psi = model_inv_psi

    def forward(self, x):
        psi_x = self.model_psi(x)
        return self.model_inv_psi(psi_x)

def Build_model(n_input, layer_sizes_dic, layer_sizes_k, n_psi_train, u_dim = 0, para = False, dic_nonlinearity = True):
    
    # DicNN model
    dic = DicNN(n_input=n_input, layer_sizes = layer_sizes_dic, n_psi_train=n_psi_train, nonlinearity = dic_nonlinearity)

    # Model Psi
    model_psi = ModelPsi(dic)

    # Model Inverse Psi
    model_inv_psi = ModelPsi(dic.inv_forward)

    # Model Autoencoder
    model_auto = ModelAutoencoder(model_psi, model_inv_psi)

    if para:
        # Koopman Layer
        k_para = ParaKoopmanLayer(layer_sizes = layer_sizes_k, input_dim = u_dim, K_dim = n_psi_train)
        
        # Model Koopman
        model_koopman = ModelParaKoopman(model_psi, k_para)

        # Model Predict
        model_predict = ModelPredictPara(model_psi, model_inv_psi, k_para)

    else:
        # Koopman Layer
        k_layer = KoopmanLayer(input_dim=n_psi_train, output_dim=n_psi_train)

        # Model Koopman
        model_koopman = ModelKoopman(model_psi, k_layer)

        # Model Predict
        model_predict = ModelPredict(model_psi, model_inv_psi, k_layer)

    
    return model_psi, model_koopman, model_inv_psi, model_predict, model_auto


class NonlinearMiddleLayer(nn.Module):
    """
    Nonlinear Middle Layer with modifications to reduce overfitting
    """

    def __init__(self, layer_sizes=[64, 64, 64], n_psi_train=128, u_dim=2, dropout_rate=0.5):
        super(NonlinearMiddleLayer, self).__init__()
        self.layer_sizes = layer_sizes

        # Input and hidden layers
        self.input_layer = nn.Linear(n_psi_train + u_dim, self.layer_sizes[0], bias=False)
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.hidden_layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            self.hidden_layers.append(nn.BatchNorm1d(layer_sizes[i]))
            self.hidden_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(layer_sizes[-1], n_psi_train)

        # Activation function
        self.activation = nn.LeakyReLU()

    def forward(self, x, u):
        y = torch.cat((x, u), dim=1)
        y = self.input_layer(y)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                y = self.activation(layer(y))
            else:
                y = layer(y)
        y = self.output_layer(y)
        return y

    
class NonlinearCorrection(nn.Module):
    def __init__(self, model_psi, model_inv_psi, middlelayer):
        super(NonlinearCorrection, self).__init__()
        self.model_psi = model_psi
        self.model_inv_psi = model_inv_psi
        self.middlelayer = middlelayer

    def forward(self, x, u):
        x_latent = self.model_psi(x)
        y_latent = self.middlelayer(x_latent, u)
        y = self.model_inv_psi(y_latent)
        return y
    
def BuildNonlinearModel(n_input, layer_sizes_dic, layer_sizes_m, n_psi_train, u_dim):
    # DicNN model
    dic = DicNN(n_input=n_input, layer_sizes = layer_sizes_dic, n_psi_train=n_psi_train, nonlinearity = True)

    # Model Psi
    model_psi = ModelPsi(dic)

    # Model Inverse Psi
    model_inv_psi = ModelPsi(dic.inv_forward)

    # Model Middle Layer
    middlelayer = NonlinearMiddleLayer(layer_sizes_m, n_psi_train, u_dim)

    # Model
    model_correction = NonlinearCorrection(model_psi, model_inv_psi, middlelayer)

    # Autoencoder
    model_autoencoder = ModelAutoencoder(model_psi, model_inv_psi)

    return model_psi, model_inv_psi, model_autoencoder, model_correction
