import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable
from datetime import datetime

import math
import random
#import tqdm
from torchtext import data, datasets, vocab

import os
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler



# Self Attention Class
class SelfAttentionConv(nn.Module):
    def __init__(self, k, headers = 8, kernel_size = 5, mask_next = True, mask_diag = False):
        super().__init__()
        
        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag
        
        h = headers
        
        # Query, Key and Value Transformations
        
        padding = (kernel_size-1)
        self.padding_opertor = nn.ConstantPad1d((padding,0), 0)
        
        self.toqueries = nn.Conv1d(k, k*h, kernel_size, padding=0 ,bias=True)
        self.tokeys = nn.Conv1d(k, k*h, kernel_size, padding=0 ,bias=True)
        self.tovalues = nn.Conv1d(k, k*h, kernel_size = 1 , padding=0 ,bias=False) # No convolution operated
        
        # Heads unifier
        self.unifyheads = nn.Linear(k*h, k)
    def forward(self, x):
        
        # Extraction dimensions
        b, t, k  = x.size() # batch_size, number_of_timesteps, number_of_time_series
        
        
        # Checking Embedding dimension
        assert self.k == k, 'Number of time series '+str(k)+' didn t much the number of k '+str(self.k)+' in the initiaalization of the attention layer.'
        h = self.headers
        
        #  Transpose to see the different time series as different channels
        x = x.transpose(1,2)
        x_padded = self.padding_opertor(x)
        
        # Query, Key and Value Transformations
        queries = self.toqueries(x_padded).view(b,k,h,t)
        keys = self.tokeys(x_padded).view(b,k,h,t)
        values = self.tovalues(x).view(b,k,h,t)
        
        # Transposition to return the canonical format
        queries = queries.transpose(1,2) # batch, header, time serie, time step (b, h, k, t)
        queries = queries.transpose(2,3) # batch, header, time step, time serie (b, h, t, k)
        
        values = values.transpose(1,2) # batch, header, time serie, time step (b, h, k, t)
        values = values.transpose(2,3) # batch, header, time step, time serie (b, h, t, k)
        
        keys = keys.transpose(1,2) # batch, header, time serie, time step (b, h, k, t)
        keys = keys.transpose(2,3) # batch, header, time step, time serie (b, h, t, k)
        
        
        # Weights 
        queries = queries/(k**(.25))
        keys = keys/(k**(.25))
        
        queries = queries.transpose(1,2).contiguous().view(b*h, t, k)
        keys = keys.transpose(1,2).contiguous().view(b*h, t, k)
        values = values.transpose(1,2).contiguous().view(b*h, t, k)
        
        
        weights = torch.bmm(queries, keys.transpose(1,2))
        
                
        ## Mask the upper & diag of the attention matrix
        if self.mask_next :
            if self.mask_diag :
                indices = torch.triu_indices(t ,t , offset=0)
                weights[:, indices[0], indices[1]] = float('-inf')
            else :
                indices = torch.triu_indices(t ,t , offset=1)
                weights[:, indices[0], indices[1]] = float('-inf')
        
        # Softmax 
        weights = F.softmax(weights, dim=2)
        
        # Output
        output = torch.bmm(weights, values)
        output = output.view(b,h,t,k)
        output = output.transpose(1,2).contiguous().view(b,t, k*h)
        
        return self.unifyheads(output) # shape (b,t,k)


# Conv Transforme Block

class ConvTransformerBLock(nn.Module):
    def __init__(self, k, headers, kernel_size = 5, mask_next = True, mask_diag = False, dropout_proba = 0.2):
        super().__init__()
        
        # Self attention
        self.attention = SelfAttentionConv(k, headers, kernel_size, mask_next, mask_diag)
        
        # First & Second Norm
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        
        # Feed Forward Network
        self.feedforward = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )
        # Dropout funtcion  & Relu:
        self.dropout = nn.Dropout(p = dropout_proba)
        self.activation = nn.ReLU()
        
    def forward(self, x, train=False):
        
        # Self attention + Residual
        x = self.attention(x) + x
        
        # Dropout attention
        if train :
            x = self.dropout(x)
        
        # First Normalization
        x = self.norm1(x)
    
        # Feed Froward network + residual
        x = self.feedforward(x) + x
        
        # Second Normalization
        x = self.norm2(x)
        
        return x


# Forcasting Conv Transformer :
class ForcastConvTransformer(nn.Module):
    def __init__(self, k, headers, depth, seq_length, kernel_size = 5, mask_next = True, mask_diag = False, dropout_proba = 0.2, num_tokens = None):
        super().__init__()
        # Embedding 
        self.tokens_in_count = False
        if num_tokens :
            self.tokens_in_count = True
            self.token_embedding = nn.Embedding(num_tokens, k)
        
        # Embedding the position
        self.position_embedding = nn.Embedding(seq_length, k)
        
        # Number of time series
        self.k = k
        self.seq_length = seq_length
        
        # Transformer blocks
        tblocks = []
        for t in range(depth):
            tblocks.append(ConvTransformerBLock(k, headers, kernel_size, mask_next, mask_diag, dropout_proba))
        self.TransformerBlocks = nn.Sequential(*tblocks)
        
        # Transformation from k dimension to numClasses
        self.topreSigma = nn.Linear(k, 1)
        self.tomu = nn.Linear(k, 1)
        self.plus = nn.Softplus()
        
    def forward(self, x, tokens = None):
        b ,t ,k = x.size()
        
        # checking that the given batch had same number of time series as the BLock had
        assert k == self.k, 'The k :'+str(self.k)+' number of timeseries given in the initialization is different than what given in the x :'+str(k)
        assert t == self.seq_length, 'The lenght of the timeseries given t '+str(t)+' miss much with the lenght sequence given in the Tranformers initialisation self.seq_length: '+str(self.seq_length)
        
        # Position embedding
        pos = torch.arange(t)
        self.pos_emb = self.position_embedding(pos).expand(b,t,k)
        
        # Checking token embedding 
        assert self.tokens_in_count == (not (tokens is None)), 'self.tokens_in_count = '+str(self.tokens_in_count)+' should be equal to (not (tokens is None)) = '+str((not (tokens is None)))
        if not (tokens is None) :
            ## checking that the number of tockens corresponde to the number of batch elements
            assert tokens.size(0) == b
            self.tok_emb = self.token_embedding(tokens)
            self.tok_emb = self.tok_emb.expand(t,b,k).transpose(0,1)
        
        # Adding Pos Embedding and token Embedding to the variable
        if not (tokens is None):
            x = self.pos_emb + self.tok_emb + x
        else:
            x = self.pos_emb + x
        
        # Transformer :
        x = self.TransformerBlocks(x)
        mu = self.tomu(x)
        presigma = self.topreSigma(x)
        sigma = self.plus(presigma)
                
        return mu, sigma

# Loading data 
class TrainDataset(Dataset):
    def __init__(self, data_path, data_name):
        print('data_name :', data_path)
        self.data = np.load(os.path.join(data_path, f'train_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'train_label_{data_name}.npy'))
        self.train_len = self.data.shape[0]
    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]), self.v[index], self.label[index])

class TestDataset(Dataset):
    def __init__(self, data_path, data_name):
        self.data = np.load(os.path.join(data_path, f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'test_v_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'test_label_{data_name}.npy'))
        self.test_len = self.data.shape[0]
    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]),self.v[index],self.label[index])

class WeightedSampler(Sampler):
    def __init__(self, data_path, data_name, replacement=True):
        v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        self.weights = torch.as_tensor(np.abs(v[:,0])/np.sum(np.abs(v[:,0])), dtype=torch.double)
        self.num_samples = self.weights.shape[0]
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples