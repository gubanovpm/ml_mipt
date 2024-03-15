import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, max_seg_length=128):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.max_seg_length = max_seg_length
        
        self.embedding = nn.Embedding(num_embeddings=input_dim,embedding_dim=emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, dropout=dropout)
        self.positional_encoding = PositionalEncoding(emb_dim, max_seg_length)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        embedded = self.embedding(src)
        embedded = self.positional_encoding(embedded)
        embedded = self.dropout(embedded)
        
        output, (hidden, cell) = self.rnn(embedded)
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, max_seg_length=128):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_seg_length = max_seg_length

        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, dropout=dropout)
        self.positional_encoding = PositionalEncoding(emb_dim, max_seg_length)
        self.out = nn.Linear(in_features=hid_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.positional_encoding(self.embedding(input)))
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        
        return prediction, hidden, cell

class PEmodel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        
        input = trg[0,:]
        
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
