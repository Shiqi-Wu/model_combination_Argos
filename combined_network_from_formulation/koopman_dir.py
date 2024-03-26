import math, copy
import numpy as np
import torch
import torch.nn as nn
from entmax import sparsemax, entmax15, entmax_bisect, entmax_bisect
import torch.nn.functional as F
from torch.autograd import Variable
# import utils

import logging
from torch.autograd import grad

from IPython import embed
import warnings
warnings.filterwarnings("ignore")

class EncoderDecoder(nn.Module):
    def __init__(self, state_encoder, state_decoder, control_encoder, state_matrix, control_matrix):
        super(EncoderDecoder, self).__init__()
        self.state_encoder = state_encoder
        self.state_decoder = state_decoder
        self.control_encoder = control_encoder
        self.state_matrix = state_matrix
        self.control_matrix = control_matrix

    def forward(self, state, control, nu):
        state_encoded = self.state_encoder(state)
        control_encoded = self.control_encoder(control)
        K_state = self.state_matrix(nu)
        K_control = self.control_matrix(nu)
        state_transformed = torch.matmul(state_encoded, K_state)
        control_transformed = torch.matmul(control_encoded, K_control)
        state_predicted = self.state_decoder(state_transformed + control_transformed)
        return state_predicted

    def encode_state(self, state):
        return self.state_encoder(state)
    
    def auto_state(self, state):
        state_encoded = self.state_encoder(state)
        state_auto = self.state_decoder(state_encoded)
        return state_auto

    def forward_latent(self, state, control, nu):
        state_encoded = self.state_encoder(state)
        control_encoded = self.control_encoder(control)
        K_state = self.state_matrix(nu)
        K_control = self.control_matrix(nu)
        state_transformed = torch.matmul(state_encoded, K_state)
        control_transformed = torch.matmul(control_encoded, K_control)
        return state_transformed + control_transformed



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6): 
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout = 0.2): 
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class FeedForwardLayerConnection(nn.Module):
    def __init__(self, size, feed_forward, dropout):
        super(FeedForwardLayerConnection, self).__init__()
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 1)
        self.size = size
    
    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.feed_forward(x))
        return x
    
class StateMatrix_sum(nn.Module):
    def __init__(self, params):
        super(StateMatrix_sum, self).__init__()
        self.k_size = params.d_model
        self.initialize_K()

    def add_nu(self, nu_data):
        for nu_tensor in nu_data:
            nu = str(nu_tensor.item())
            if nu not in self.k_matrices:
                self.k_matrices[nu] = nn.Parameter(torch.randn(self.k_size, self.k_size))
    
    def initialize_K(self):
        self.k_matrices = nn.ModuleDict()

    def forward(self, nu):
        nu = str(nu.item())
        if nu in self.k_matrices:
            return self.k_matrices[nu]
        else:
            raise KeyError(f"nu value '{nu}' not found in k_layers.")

    def forward_lambda(self, lbd):
        if len(lbd) != len(self.k_matrices):
            raise ValueError("Length of lambda does not match number of matrices in k_matrices.")

        K_sum = torch.zeros((self.k_size, self.k_size))
        for i, (k_matrix, lbd_i) in enumerate(zip(self.k_matrices, lbd)):
            K_sum += lbd_i * k_matrix
        return K_sum
        

class StateMatrix_NN(nn.Module):
    def __init__(self, params):
        super(StateMatrix_NN, self).__init__()
        self.input_layer = nn.Linear(params.nu_dim, params.k_model)
        self.MatrixLayer = FeedForwardLayerConnection(params.k_model, FeedForward(params.k_model, params.k_model), params.dropout)
        self.layers = clones(FeedForwardLayerConnection, params.N_Koopman)
        self.norm = LayerNorm(params.k_model)
        self.output_layer = nn.Linear(params.k_model, params.d_model * params.d_model)
    
    def forward(self, nu):
        x = F.relu(self.input_layer(nu))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_layer(x)
        x = x.reshape(-1, 1, self.params.d_model) 
        return x

class ControlEncoder(nn.Module):
    def __init__(self, params):
        super(ControlEncoder, self).__init__()
        self.input_layer = nn.Linear(params.u_dim, params.u_model)
        self.Layer = FeedForwardLayerConnection(params.u_model, FeedForward(params.u_model, params.u_model), params.dropout)
        self.layers = clones(self.Layer, params.N_Control)
        self.norm = LayerNorm(params.u_model)
        self.output_layer = nn.Linear(params.u_model, params.d_model)
    
    def forward(self, control):
        x = F.relu(self.input_layer(control))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_layer(x)
        return x

class StateEncoder(nn.Module):
    def __init__(self, params):
        super(StateEncoder, self).__init__()
        self.params = params
        self.input_layer = nn.Linear(params.x_dim, params.d_model)
        attn = MultiHeadedAttention(params)
        ff = FeedForward(params.d_model, params.d_ff)
        self.Layer = StateEncoderDecoderLayer(params.d_model, attn, ff, params.dropout)
        self.layers = clones(self.Layer, params.N_State)
        self.norm = LayerNorm(params.d_model)

    def forward(self, state):
        state = torch.reshape(state, (-1, 1, self.params.x_dim))
        x = F.relu(self.input_layer(state))
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
        
class StateDecoder(nn.Module):
    def __init__(self, params):
        super(StateDecoder, self).__init__()
        self.params = params
        self.input_layer = nn.Linear(params.d_model, params.d_model)
        attn = MultiHeadedAttention(params)
        ff = FeedForward(params.d_model, params.d_ff)
        self.Layer = StateEncoderDecoderLayer(params.d_model, attn, ff, params.dropout)
        self.layers = clones(self.Layer, params.N_State)
        self.norm = LayerNorm(params.d_model)
        self.output_layer = nn.Linear(params.d_model, params.x_dim)
    
    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_layer(x)
        return torch.reshape(x, (-1, self.params.x_dim))

class StateEncoderDecoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(StateEncoderDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    def __init__(self, params, dropout=0.2):  # TODO : h , dropout
        "Take in model size and number of heads." 
        super(MultiHeadedAttention, self).__init__()
        assert params.d_model % params.h == 0

        self.d_k = params.d_model // params.h
        self.h = params.h
        self.linears = clones(nn.Linear(params.d_model, params.d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.params = params
        self.scores = None
        # self.alpha_choser = AlphaChooser(params.h)
        self.alpha = None
        self.attn_type = params.attn_type
    
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        if self.attn_type=='entmax':
            self.alpha = self.alpha_choser()
        x, self.scores, self.attn = attention(query, key, value, self.params, mask=mask, 
                                     dropout=self.dropout, alpha=self.alpha)
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    
def attention(query, key, value, params, mask=None, dropout=None, alpha=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        try:
            scores = scores.masked_fill(mask == 0, -1e9)
        except:
            embed()

    if params.attn_type=='softmax':
        p_attn = F.softmax(scores, dim = -1)
    elif params.attn_type=='sparsemax':
        p_attn = sparsemax(scores, dim=-1)
    elif params.attn_type=='entmax15':
        p_attn = entmax15(scores, dim=-1)
    elif params.attn_type=='entmax':
        p_attn = entmax_bisect(scores, alpha, n_iter=25)
    else:
        raise Exception
    if dropout is not None:
        p_attn = dropout(p_attn)
    p_attn = p_attn.to(torch.float32)
    return torch.matmul(p_attn, value), scores, p_attn

def BuildModelFromParams(params):
    state_encoder = StateEncoder(params)
    control_encoder = ControlEncoder(params)
    state_matrix = StateMatrix_sum(params)
    control_matrix = StateMatrix_sum(params)
    state_decoder = StateDecoder(params)
    model = EncoderDecoder(state_encoder, state_decoder, control_encoder, state_matrix, control_matrix)
    return model

