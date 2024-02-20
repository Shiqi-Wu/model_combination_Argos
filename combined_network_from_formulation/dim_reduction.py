import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dim, dropout=0.2):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encode_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dim)-1):
            self.encode_hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                nn.Dropout(dropout)
            ))
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(nn.Linear(input_dim, hidden_dim[0]))
        for layer in self.encode_hidden_layers:
            self.encoder_layers.append(layer)
        self.encoder_layers.append(nn.Linear(hidden_dim[-1], encoding_dim))
        
        # Decoder layers
        self.decode_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dim)-1, 0, -1):
            self.decode_hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i-1]),
                nn.Dropout(dropout)
            ))
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(encoding_dim, hidden_dim[-1]))
        for layer in self.decode_hidden_layers:
            self.decoder_layers.append(layer)
        self.decoder_layers.append(nn.Linear(hidden_dim[0], input_dim))
        
        # Encoder and Decoder
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self,x):
        return self.decoder(x)

def build_model(input_dim, encoding_dim, hidden_dim):
    model = Autoencoder(input_dim, encoding_dim, hidden_dim)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    print(f"Training Epoch: {epoch + 1}")
    for inputs, _, _, _ in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(dataloader):.4f}")
    return running_loss / len(dataloader)

def test_one_epoch(model, dataloader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, _, _, _ in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Testing Loss: {running_loss / len(dataloader):.4f}")
    return running_loss / len(dataloader)