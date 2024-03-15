import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import torch.nn.functional as F

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

class Attention(nn.Module):
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        # self.v = nn.Parameter(torch.rand(hidden_size * 2))
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: from decoder, [batch, decoder_hidden_size]
        timestep = encoder_outputs.shape[0]
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)    # [batch, timestep, decoder_hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)    # [batch, timestep, encoder_hidden_size]
        
        # [batch, timestep]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)    # [batch, 1, timestep]
         
    def score(self, hidden, encoder_outputs):
        # hidden: [batch, timestep, decoder_hidden_size]
        # encoder_outputs: [batch, timestep, encoder_hidden_size]
        # energy: [batch, timestep, hidden_size]
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)    # [batch, 2 * hidden_size, timestep]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)    # [batch, 1, 2 * hidden_size]
        energy = torch.bmm(v, energy)    # [batch, 1, timestep]
        return energy.squeeze(1)    # [batch, timestep]

class Multi_head_attention_trs(nn.Module):
    def __init__(self, hidden_size, nhead=4, dropout=0.3):
        super(Multi_head_attention_trs, self).__init__()
        self.nhead = nhead
        self.hidden_size = hidden_size
        
        if hidden_size % nhead != 0:
            raise Exception(f'hidden_size must be divisble by nhead, but got {hidden_size}/{nhead}.')
        
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, nhead)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.final_attn = Attention(hidden_size)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden]
        # encoder_outputs: [seq, batch, hidden]
        # return: context [1, batch, seq]
        
        # context: [seq, batch, hidden]
        context, _ = self.multi_head_attention(encoder_outputs, 
                                               encoder_outputs, 
                                               encoder_outputs)
        context = context + encoder_outputs
        context = torch.tanh(self.layer_norm(context))
        attn_weights = self.final_attn(hidden.unsqueeze(0), context)
        context = attn_weights.bmm(context.transpose(0, 1))
        context = context.transpose(0, 1)
        return context

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, max_seg_length=128):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.max_seg_length = max_seg_length

        self.layers = nn.ModuleDict({
            'embedding' : nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim),
            'rnn'       : nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, dropout=dropout),
            'pos_encod' : PositionalEncoding(emb_dim, max_seg_length),
            'dropout'   : nn.Dropout(p=dropout)})

    def forward(self, src):
        embedded = self.layers['pos_encod'](self.layers['embedding'](src))
        embedded = self.layers['dropout'](embedded)

        output, (hidden, cell) = self.layers['rnn'](embedded)

        return output, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, max_seg_length=128):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.layers = nn.ModuleDict({
            'embedding' : nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim),
            'pos_encod' : PositionalEncoding(emb_dim, max_seg_length),
            'rnn'       : nn.LSTM(input_size=emb_dim+hid_dim, hidden_size=hid_dim, num_layers=n_layers, dropout=dropout),
            'linear'    : nn.Linear(in_features=hid_dim, out_features=output_dim),
            'attention' : Multi_head_attention_trs(hidden_size=hid_dim, dropout=dropout),
        })

    def forward(self, inp, enc_hidden, enc_output, cell, device):
        embedded = self.layers['pos_encod'](self.layers['embedding'](inp.unsqueeze(0)))
        key = enc_hidden.sum(axis=0)
        context = self.layers['attention'](key, enc_output)

        rnn_input = torch.cat([embedded, context], 2)

        output, (hidden, cell) = self.layers['rnn'](rnn_input, (enc_hidden, cell))
        output = self.layers['linear'](output.squeeze(0))
        output = F.log_softmax(output, dim=1)

        return output, hidden, cell

class Megazord(nn.Module):
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
        batch_size, max_len = src.shape[1], trg.shape[0]
        
        outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim).to(self.device)

        if torch.cuda.is_available():
            outputs = outputs.cuda()
        
        # encoder_output: [seq_len, batch, hidden_size]
        # hidden: [1, batch, hidden_size]
        encoder_output, hidden, cell = self.encoder(src)
        # hidden = hidden[-1:]
        output = trg[0, :]
        
        use_teacher = random.random() < teacher_forcing_ratio
        if use_teacher:
            for t in range(1, max_len):
                output, hidden, cell = self.decoder(output, hidden, encoder_output, cell, self.device)
                outputs[t] = output
                output = trg[t]
        else:
            for t in range(1, max_len):
                output, hidden, cell = self.decoder(output, hidden, encoder_output, cell, self.device)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        
        # [max_len, batch, output_size]
        return outputs

