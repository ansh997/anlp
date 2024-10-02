import torch
import torch.nn as nn
import math
import re

def regex_tokenizer(sentence):
    return re.findall(r'\b\w+\b', sentence.lower())

def tokenize(self, sentence):
    return [self.word2idx.get(word, self.unk_idx) for word in regex_tokenizer(sentence)]



class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout=0.1) -> None:
        super(PositionalEncoding, self).__init__()  # why this?
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even terms
        pe[:, 1::2] = torch.cos(position * div_term)  # odd terms
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # allows the buffer to be automatically moved to the device (CPU or GPU)
        
    def forward(self, x):
        # print('\tinside Positional Encoding: ', x.size(), self.pe.size())
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)  # check this while debugging
        return self.dropout(x)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
    
# Add more tokenizers, right now only simple_tokenizer is implemented

def simple_tokenizer(sentence):
    sentence = re.sub(r'<[^>]+>', '', sentence)
    return sentence.lower().split()
    

