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
    def create_node(self, dropout):
        return nn.ModuleDict({
            'SelfAttention'  : nn.MultiheadAttention(embed_dim=self.emb_dim, num_heads=2, dropout=dropout),
            'LayerNorm_1'    : nn.LayerNorm(self.emb_dim),
            'CrossAttention' : nn.MultiheadAttention(embed_dim=self.emb_dim, num_heads=2, dropout=dropout),
            'LayerNorm_2'    : nn.LayerNorm(self.emb_dim),
            'Linear_1'       : nn.Linear(in_features=self.emb_dim, out_features=self.hid_dim),
            'Activation'     : nn.ReLU(self.emb_dim),
            'Dropout'        : nn.Dropout(p=dropout),
            'Linear_2'       : nn.Linear(in_features=self.hid_dim, out_features=self.emb_dim),
            'LayerNorm_3'    : nn.LayerNorm(self.emb_dim)})

    def __init__(self, emb_dim, n_layers, dropout, hid_dim, input_dim, max_seg_legth=128):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim, max_seg_legth)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.nodes = nn.ModuleList([self.create_node(dropout) for _ in range (self.n_layers)])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, src, src_mask):
        inp = self.dropout_1(self.positional_encoding(self.embedding(src)))

        for node in self.nodes:
            t, _  = node['SelfAttention'](inp, inp, inp, src_mask)
            t     = node['LayerNorm_1'](torch.add(t, inp))

            inp = node['Linear_1'](t)
            inp = node['Activation'](inp)
            inp = node['Dropout'](inp)
            inp = node['Linear_2'](inp)
            inp = self.dropout_2(inp)

            inp = node['LayerNorm_2'](torch.add(inp, t))

        return inp

class Decoder(nn.Module):
    def create_node(self, dropout):
        return nn.ModuleDict({
            'SelfAttention'  : nn.MultiheadAttention(embed_dim=self.emb_dim, num_heads=2, dropout=dropout),
            'LayerNorm_1'    : nn.LayerNorm(self.emb_dim),
            'CrossAttention' : nn.MultiheadAttention(embed_dim=self.emb_dim, num_heads=2, dropout=dropout),
            'LayerNorm_2'    : nn.LayerNorm(self.emb_dim),
            'Linear_1'       : nn.Linear(in_features=self.emb_dim, out_features=self.hid_dim),
            'Activation'     : nn.ReLU(self.emb_dim),
            'Dropout'        : nn.Dropout(p=dropout),
            'Linear_2'       : nn.Linear(in_features=self.hid_dim, out_features=self.emb_dim),
            'LayerNorm_3'    : nn.LayerNorm(self.emb_dim)})

    def __init__(self, output_dim, emb_dim, n_layers, hid_dim, dropout, max_seg_legth=128):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_embeddings=output_dim,embedding_dim=emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim, max_seg_legth)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.nodes = nn.ModuleList([self.create_node(dropout) for _ in range(self.n_layers)])
        self.out = nn.Linear(in_features=emb_dim, out_features=output_dim)
        # self.softmax = nn.Softmax(output_dim)

    def forward(self, inp, encode_inp, src_mask, trg_mask):
        inp = self.dropout_1(self.positional_encoding(self.embedding(inp.unsqueeze(0))))

        for node in self.nodes : 
            t, _ = node['SelfAttention'](inp, inp, inp, trg_mask)
            t = node['LayerNorm_1'](torch.add(t, inp))
            
            output, _ = node['CrossAttention'](query=t, key=encode_inp, value=encode_inp, key_padding_mask=src_mask)
            output = node['LayerNorm_2'](torch.add(t, output))
            
            inp = node['Linear_1'](output)
            inp = node['Activation'](inp)
            inp = node['Dropout'](inp)
            inp = node['Linear_2'](inp)
            inp = self.dropout_2(inp)

            inp = node['LayerNorm_3'](torch.add(output, inp))

        output = self.out(inp.squeeze(0))
        # output = self.softmax(output)
        return output

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, device, input_dim, output_dim, emb_dim_enc, emb_dim_dec, src_pad_idx, trg_pad_idx, trg_sos_idx):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.emb_dim_enc = emb_dim_enc
        self.emb_dim_dec = emb_dim_dec
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        src_mask = None#self.make_src_mask(src)
        trg_mask = None#self.make_trg_mask(trg)

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        enc_output = self.encoder(src, src_mask)
        # print(enc_output.shape)
        inp = trg[0, :]

        for t in range(1, max_len):
            output = self.decoder(inp, enc_output, src_mask, trg_mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            inp = (trg[t] if teacher_force else top1)
        
        return outputs