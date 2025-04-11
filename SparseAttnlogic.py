from select import select
from time import sleep
from turtle import forward
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from math import pi
import torch.nn.functional as F
from scipy import signal
from torchsparseattn import fused
#import numpy as np

signal_size = 1024
batch_size = 2
MAX_LIMIT =10.000**6


def get_tensor_info(tensor):
  info = []
  for name in ['requires_grad', 'is_leaf', 'grad_fn','grad']:
    info.append(f'{name}({getattr(tensor, name)})')
  info.append(f'tensor({str(tensor)})')
  return ' '.join(info)

def compute_periodogram(signals):
    """
    Compute the periodogram for a batch of multi-channel signals.

    Parameters:
    - signals (torch.Tensor): A tensor of shape (batch_size, num_channels, signal_length)

    Returns:
    - torch.Tensor: Periodogram for each signal in each channel
    """
    # Compute the Fourier Transform along the last dimension
    fft_results = torch.fft.fft(signals, dim=-1)

    # Compute the Power Spectrum (magnitude squared)
    power_spectrum = torch.abs(fft_results) ** 2 / signals.size(-1)

    # Typically, we use the positive frequencies for periodograms
    return power_spectrum[..., :signals.size(-1) // 2]

def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 50
    w = 2 * pi * f
    q = torch.tensor(1 - pow(ep, 2))
    #print('laplase',p)
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))
    return y

def hlogic(z):
    # First, clamp the values of z to be between 0 and 1
    #clamped_z = torch.clamp(z, min=0, max=1)
    # Then apply ReLU to ensure all values are non-negative (this step is actually redundant here
    # since clamping between 0 and 1 already ensures all values are >= 0)
    clamped_z = torch.relu(z)
    #result = torch.(clamped_z)
    return clamped_z

class LaplacePredicate(nn.Module):
    def __init__(self, out_channels, kernel_size, in_channels=1):
        super(LaplacePredicate, self).__init__()
        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
            
        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels,dtype=torch.float32)).view(-1, 1)
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels,dtype=torch.float32)).view(-1, 1)
        self.bais = nn.Parameter(torch.randn(out_channels))
 
        

    def forward(self, waveforms):
        #print(self.filters.device)
 
        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size))).to(waveforms.device)
        self.b_ = self.b_.to(time_disc.device)
        self.a_ = self.a_.to(time_disc.device)
        self.bais = self.bais.to(time_disc.device)
        p1 = time_disc - self.b_ / self.a_
        laplace_filter = Laplace(p1)
        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.filters = self.filters.to(self.device)  # Move to the correct device
        result = F.conv1d(waveforms, self.filters, bias=self.bais, stride=1, padding=16)
        return result
    
class PeriodogramPredicate(nn.Module):
    def __init__(self, fs,nfft,bais=1,out_channels=10):
        super(PeriodogramPredicate,self).__init__()    

        self.nfft = nfft
        self.fs  = fs
        self.weight = nn.Parameter(torch.randn(out_channels,1,dtype=torch.float32))
        self.bais =  bais
 

    def forward(self,input):
        Pxx_den = compute_periodogram(input)
    
        x = nn.functional.normalize(torch.tensor(Pxx_den).to(input.device),p=2,dim=-1)
        if self.bais>0:
            return x-self.weight
        else:

            return self.weight-x

# -----------------------input size>=111---------------------------------

class FCAlwaysLayer(nn.Module):
    def __init__(self, input_size, output_size, max_limit=1e6):
        super(FCAlwaysLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(output_size, input_size,dtype=torch.float32))
        self.beta = nn.Parameter(torch.randn(1,dtype=torch.float32))
        self.max_limit = max_limit

    def forward(self, x):
        # Calculate the weighted product for each output dimension
        # Handle two-dimensional input (batch_size, input_size)
        # Expand weights to match the batch size
        weights_expanded = self.weights.unsqueeze(0).expand(x.shape[0], -1, -1)

        # Calculate the weighted product for each output dimension
        wrho = (1-x.unsqueeze(1)) * weights_expanded
        row_sums = self.beta-wrho.sum(dim=2)
        output = hlogic(row_sums)
        return output

class FCEventuallyLayer(nn.Module):
    def __init__(self, input_size, output_size, max_limit=1e6):
        super(FCEventuallyLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(output_size, input_size,dtype=torch.float32))
        self.max_limit = max_limit
        self.beta = nn.Parameter(torch.randn(1,dtype=torch.float32))

    def forward(self, x):
        # Calculate the weighted product for each output dimension
        # Handle two-dimensional input (batch_size, input_size)
        # Expand weights to match the batch size
        weights_expanded = self.weights.unsqueeze(0).expand(x.shape[0], -1, -1)

        # Calculate the weighted product for each output dimension
        wrho = x.unsqueeze(1) * weights_expanded
        #wrho = torch.prod(weighted_input, dim=1).double()
        # Eventually function logic
        row_sum = wrho.sum(dim=2)
        output = hlogic(1-self.beta+row_sum)
 
        return output

class ResFCEventuallyLayer(nn.Module):
    def __init__(self, output_size, max_limit=1e6):
        super(ResFCEventuallyLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(output_size, output_size,dtype=torch.float32))
        self.max_limit = max_limit
        self.beta = nn.Parameter(torch.randn(1,dtype=torch.float32))
        self.res = ResEventually(output_size)

    def forward(self, x):
        # Calculate the weighted product for each output dimension
        # Handle two-dimensional input (batch_size, input_size)
        # Expand weights to match the batch size
        weights_expanded = self.weights.unsqueeze(0).expand(x.shape[0], -1, -1)

        # Calculate the weighted product for each output dimension
        wrho = x.unsqueeze(1) * weights_expanded
        #wrho = torch.prod(weighted_input, dim=1).double()
        # Eventually function logic
        row_sum = wrho.sum(dim=2)
        output = hlogic(1-self.beta+row_sum)
        output = self.res(x,output)
 
        return output

class Conv1DAlwaysLayer(nn.Module):
    def __init__(self, channels, kernel_size,step =1, MAX_LIMIT = 1e6):
        super(Conv1DAlwaysLayer, self).__init__()
        self.kernel_size = kernel_size
        self.MAX_LIMIT = MAX_LIMIT
        self.step = step
        # Initialize weights for each channel separately
        self.weights = nn.Parameter(torch.randn(channels, kernel_size,dtype=torch.float32))
        self.beta = nn.Parameter(torch.randn(channels,dtype=torch.float32))
        self.leakrulu = nn.LeakyReLU(0.02)

 

    def forward(self, x):
        # Unfold x to create windows
        # Initial windows shape: [batch_size, channels, kernel_size, num_windows]
        windows = x.unfold(dimension=2, size=self.kernel_size, step=self.step)
        # Permute windows to shape [batch_size, channels, num_windows, kernel_size]
        # to align with weights_adjusted for broadcasting
        windows_permuted = windows.permute(0, 1, 3, 2)
        # weights_adjusted shape: [1, channels, kernel_size, 1] for broadcasting
        weights_adjusted = self.weights.unsqueeze(0).unsqueeze(-1)
        # Perform element-wise multiplication
        # This now correctly aligns windows_permuted with weights_adjusted for broadcasting
        weighted_windows = (1-windows_permuted) * weights_adjusted

        # Since the shape of weighted_windows is now [batch_size, channels, num_windows, kernel_size],
        # and you may want to apply 'eventually' across the last dimension,
        # you should reshape or permute as necessary before applying 'eventually'
        weighted_windows_reshaped = weighted_windows.permute(0, 1, 3, 2).contiguous()
        row_sums = self.beta.view(1, -1, 1)-weighted_windows_reshaped.sum(dim=3)
        results = hlogic(row_sums)
        results = self.leakrulu(results)
        #print('the size of the output from conv1d',results.size())
        return results



class Conv1DEventuallyLayer(nn.Module):
    def __init__(self, channels, kernel_size,step = 1, max_limit=1e6):
        super(Conv1DEventuallyLayer, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.max_limit = max_limit
        self.step  = step
        # Initialize weights as a trainable parameter
        # Shape: [channels, kernel_size] assuming weights are applied per-channel
        self.weights = nn.Parameter(torch.randn(channels, kernel_size,dtype=torch.float32))
        self.beta = nn.Parameter(torch.randn(channels,dtype=torch.float32))
        self.leakrulu = nn.LeakyReLU(0.02)
        

 
    def forward(self, x):
        # Unfold x to create windows
        # Initial windows shape: [batch_size, channels, kernel_size, num_windows]
        windows = x.unfold(dimension=2, size=self.kernel_size, step=self.step)
        # Permute windows to shape [batch_size, channels, num_windows, kernel_size]
        # to align with weights_adjusted for broadcasting
        windows_permuted = windows.permute(0, 1, 3, 2)
        # weights_adjusted shape: [1, channels, kernel_size, 1] for broadcasting
        weights_adjusted = self.weights.unsqueeze(0).unsqueeze(-1)
        # Perform element-wise multiplication
        # This now correctly aligns windows_permuted with weights_adjusted for broadcasting
        #print(windows_permuted.size(),weights_adjusted.size())
        weighted_windows = windows_permuted * weights_adjusted

        # Since the shape of weighted_windows is now [batch_size, channels, num_windows, kernel_size],
        # and you may want to apply 'eventually' across the last dimension,
        # you should reshape or permute as necessary before applying 'eventually'
        weighted_windows_reshaped = weighted_windows.permute(0, 1, 3, 2).contiguous()
        
        row_sum = self.beta.view(1, -1, 1)-weighted_windows_reshaped.sum(dim=3)
        results = hlogic(1-row_sum)
        results = self.leakrulu(results)
        #print('the output size is:',results.size())
        return results
class ResEventually(nn.Module):
    def __init__(self, input_size=16):
        super(ResEventually, self).__init__()
        self.beta = nn.Parameter(torch.randn(input_size,dtype=torch.float32))
        self.leakrulu = nn.LeakyReLU(0.02)    

    def forward(self, inputx,inputy):    
        #inputx = inputx.unsqueeze(1)
        #inputy = inputy.unsqueeze(1)
        input =  inputx+inputy
        row_sum = self.beta-input
        results = hlogic(1-row_sum)
        results = self.leakrulu(results)
        #print('the output size is:',results.size())
        return results

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(torch.log(torch.tensor([10000.0],device=device)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.encoding[:x.size(0), :]


class TransformerEncoder(nn.Module):
    def __init__(self, feature_size, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(feature_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, feature_size, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.pos_decoder = PositionalEncoding(feature_size)
        decoder_layers = nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)

    def forward(self, tgt, memory):
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        return output

class TransformerAutoencoder(nn.Module):
    def __init__(self, feature_size, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = TransformerEncoder(feature_size, nhead, num_encoder_layers, dropout)
        self.decoder = TransformerDecoder(feature_size, nhead, num_decoder_layers, dropout)

    def forward(self, x):
        # Permute input to [seq_length, batch_size, feature_size] for Transformer
        x = x.permute(1, 0, 2)
        encoded = self.encoder(x)
        decoded = self.decoder(x, encoded)
        # Permute output back to [batch_size, seq_length, feature_size]
        decoded = decoded.permute(1, 0, 2)
        return decoded


class TransformerAutoencoders(nn.Module):
    def __init__(self, seq_len, d_model=512, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, include_pos_embedding=True):
        super(TransformerAutoencoders, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.include_pos_embedding = include_pos_embedding
        
        # Time step embedding: Embeds each scalar in the sequence independently
        self.time_step_embedding = nn.Linear(1, d_model)
        
        # Positional Encoding: Optional and added to the embeddings if enabled
        if include_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        # Output layer: Projects from d_model back to the original scalar value
        self.output_layer = nn.Linear(d_model, 1)
 

    def forward(self, x):
        # Reshape input x to have a 'feature' dimension of 1
        x = x.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]     
        # Embedding each time step
        embedded = self.time_step_embedding(x)  # Shape: [batch_size, seq_len, d_model]        
        if self.include_pos_embedding:
            embedded += self.pos_embedding[:, :self.seq_len, :]
        # Encoder
        encoded = self.transformer_encoder(embedded)
        # Decoder
        decoded = self.transformer_decoder(encoded, encoded)
        # Project back to original scalar value per time step
        out = self.output_layer(decoded).squeeze(-1)  # Removing feature dimension, shape: [batch_size, seq_len]
        return out
    


 
class Sparse_Attn_Logic(nn.Module):

    def __init__(self, in_channel=1, out_channel=10, num_channles =32):
        super(Sparse_Attn_Logic, self).__init__()
        self.predicate_pos = nn.Sequential(
            LaplacePredicate(num_channles, 16),
            nn.BatchNorm1d(num_channles),
            #nn.Sigmoid(),
            nn.AvgPool1d(3,2),
        )   
        self.fuseautoencoder = nn.Sequential(
            TransformerAutoencoder(feature_size=520),
            nn.AdaptiveAvgPool1d(520),
            #fused.Fusedmax(0.001),   
            nn.BatchNorm1d(num_channles),
            nn.Sigmoid(), 
            fused.Fusedmax3D(0.1), 
                           )
        self.norm = nn.BatchNorm1d(num_channles)
 
        self.conv1d = nn.Sequential(
            #nn.MaxPool1d(kernel_size=3, stride=3),
            Conv1DAlwaysLayer(num_channles,8,step=2),
            Conv1DEventuallyLayer(num_channles,8,step =2),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(num_channles),
            Conv1DEventuallyLayer(num_channles,8,step =2),
            nn.Dropout(0.4),            
            #nn.AvgPool1d(5,5),
        )
        self.conv1dFG = nn.Sequential(
            #nn.MaxPool1d(kernel_size=6, stride=3),
            Conv1DEventuallyLayer(num_channles,8,step=2),
            Conv1DAlwaysLayer(num_channles,8,step =2),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.BatchNorm1d(num_channles),
            Conv1DEventuallyLayer(num_channles,8,step =2),
            nn.Dropout(0.4),
            #nn.AvgPool1d(5,5),
        )

        self.fc = nn.Sequential(
            nn.Linear(768,1024),        # linear layer is equal to eventually layer, since the semantic is the same
            nn.ReLU(inplace=False),
            nn.Linear(1024,512),
            nn.Dropout(),
            nn.ReLU(inplace=False),
            nn.Linear(512,128),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(128,out_channel),
            nn.ReLU(inplace=False),
            
        ) 

 
    def forward(self, x):
        #print(x.is_cuda)
        x  = self.predicate_pos(x)
        weight = self.fuseautoencoder(x)
        output = weight*x
        output = self.norm(output)
        x1 =self.conv1d(x)
        x2 = self.conv1dFG(x)
        x = torch.cat((x1,x2),dim=1)

        x = x.float()  # Convert x to float

        x = x.view(x.shape[0],-1)
        #print('dtype to fc', x[0])
        out =self.fc(x)
        #print('dtype to out', out[0])
        return out


"""  
# Parameters for the model
in_channel = 16           # Length of the input/output sequence
out_channel = 10          # Dimension of the model
num_pieces = 20            # Number of heads in multiheadattention

model = Sparse_Attn(in_channel,out_channel,num_pieces)
# Example input (batch_size, seq_len)
example_input = torch.randn(16,1, 1024)

output = model(example_input)

print(f"Input Shape: {example_input.shape}")
print(f"Reconstructed Output Shape: {output.shape}") """