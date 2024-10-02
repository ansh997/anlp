import torch
import torch.nn as nn
import torch
from utils import PositionalEncoding
import math

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads) -> None:
#         super(MultiHeadAttention, self).__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
        
#         self.d_k = d_model // num_heads   # for multi-head attention, d_model should be divisible by num_heads
        
#         assert self.d_k * num_heads == d_model, "d_model should be divisible by num_heads"
        
#         self.W_Q = nn.Linear(d_model, d_model)  #query
#         self.W_K = nn.Linear(d_model, d_model)  #key
#         self.W_V = nn.Linear(d_model, d_model)  #value        
#         self.W_O = nn.Linear(d_model, d_model)  #output
        
#         # add scale too for the scaled dot product attention
#         self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
        
#     def forward(self, query, key, value, mask=None):
#         # query = [batch_size, query_len, d_model]; same for key and value
        
#         B, L, D = query.shape
        
#         Q = self.W_Q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, L, d_k]
#         K = self.W_K(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, L, d_k]
#         V = self.W_V(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, L, d_k]
        
#         # Make sure all tensors are on the same device
#         Q = Q.to(query.device)
#         K = K.to(query.device)
#         V = V.to(query.device)
#         self.scale = self.scale.to(query.device)
    
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, L, L]
        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e10)  # mask is [B, 1, 1, L]
        
#         attention = torch.softmax(scores, dim=-1)  # [B, num_heads, L, L]
        
#         x = torch.matmul(attention, V).transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        
#         return self.W_O(x)  # [B, L, D]
    

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float=0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model # Embedding vector size
        self.h = num_heads # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % num_heads == 0, "d_model is not divisible by h"

        self.d_k = d_model // num_heads # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear2(self.dropout(self.relu(self.linear1(x))))
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x = [B, L, D]
        # mask = [B, 1, 1, L]
        
        # First sub-layer - multi-head attention & Add and Norm
        attention = self.multi_head_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attention))
                
        # Second sub-layer - feed forward & Add and Norm
        feed_forward = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(feed_forward))
        
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.pe = PositionalEncoding(d_model, max_seq_len, dropout)
        
    def forward(self, src, src_mask=None):
        # x = [B, L, D]
        # mask = [B, 1, 1, L]
        
        src = self.pe(src)  # input_embedding + positional_encoding
        
        for layer in self.layers:
            src = layer(src, src_mask)  # a encoder layer is a stack of multi-head attention and feed forward with Add and Norm
        
        return src
        



