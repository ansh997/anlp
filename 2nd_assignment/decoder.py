import torch.nn as nn
from encoder import MultiHeadAttention, FeedForward
from utils import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        # x = [B, L, D]
        # enc_out = [B, L, D]
        # src_mask = [B, 1, 1, L]
        # tgt_mask = [B, 1, L, L]
        
        # First sub-layer - self attention & Add and Norm
        self_attention = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.layer_norm1(tgt + self.dropout(self_attention))
        
        # Second sub-layer - cross attention & Add and Norm
        cross_attention = self.cross_attention(tgt, enc_out, enc_out, src_mask)  # cross attention with encoder output - done correctly
        tgt = self.layer_norm2(tgt + self.dropout(cross_attention))
        
        # Third sub-layer - feed forward & Add and Norm
        feed_forward = self.feed_forward(tgt)
        tgt = self.layer_norm3(tgt + self.dropout(feed_forward))
        
        return tgt


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1) -> None:
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.pe = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.fc_out = nn.Linear(d_model, vocab_size)  # lst output layer is a linear layer of vocab_size
        
    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        # x = [B, L, D]
        # enc_out = [B, L, D]
        # src_mask = [B, 1, 1, L]
        # tgt_mask = [B, 1, L, L]
        # print("Shape of tgt:", tgt.shape)  # Add this line
        
        tgt = self.pe(tgt)  # input_embedding + positional_encoding
        
        # print("Shape of tgt:", tgt.shape)  # Add this line
        # print("Shape of enc_out:", enc_out.shape)  # Add this line
        
        for layer in self.layers:
            tgt = layer(tgt, enc_out, src_mask, tgt_mask)  # a decoder layer is a stack of self attention, cross attention and feed forward with Add and Norm
        
        # output layer is a linear layer of vocab_size with softmax activation function to output probabilities
        return self.fc_out(tgt).softmax(dim=-1) 
