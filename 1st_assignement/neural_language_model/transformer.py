import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
import re
import math
from collections import Counter

scratch_location = f'/scratch/hmnshpl/anlp_data'
filename = 'Auguste_Maquet.txt'
emb_filename = 'glove.6B.300d.txt'

data_filepath = os.path.join(scratch_location, filename)
emb_filepath = os.path.join(scratch_location, emb_filename)


def get_auguste_maquet_corpus(file_path = None):
    if filename is None:
        url = "https://www.gutenberg.org/files/7849/7849-0.txt"
        response = requests.get(url)
        text = response.text
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
    # Remove header and footer
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    text = text[start:end]
    # Clean text
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def tokenize(text):
    return text.split()

class Vocabulary:
    def __init__(self, tokens):
        self.itos = ["<unk>", "<pad>", "<sos>", "<eos>"] + list(set(tokens))
        self.stoi = {token: i for i, token in enumerate(self.itos)}
    
    def __len__(self):
        return len(self.itos)
    
    def encode(self, tokens):
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]
    
    def decode(self, ids):
        return [self.itos[id] for id in ids]
    

class TextDataset(Dataset):
    def __init__(self, text, vocab, seq_length):
        self.text = text
        self.vocab = vocab
        self.seq_length = seq_length
        self.tokens = tokenize(self.text)
        self.data = self.vocab.encode(self.tokens)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_length])
        y = torch.tensor(self.data[idx+1:idx+self.seq_length+1])
        return x, y
    

# 3. Model architecture
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tgt, tgt_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, tgt, tgt_mask)
        return self.fc_out(output)
    
    def calculate_loss(self, output, target):
        return self.criterion(output.view(-1, output.size(-1)), target.view(-1))
    
    
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


