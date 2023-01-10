import math
from typing import Tuple
import os
from random import shuffle
import copy
import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    # ntoken: size of vocabulary
    # nclasses: number of classes
    # d_model: embedding dimension
    # nhead: number of heads for multiheaded attention
    # d_hid: dimension of the ffnn in nn.TransformerEncoder
    # nlayers: number of layers in Transformer Encoder 
    # dropout: dropout probability

    def __init__(self, ntoken: int, nclasses: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, nclasses)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        #print(output.shape)
        output = output.mean(dim=0)
        #print(output.shape)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_from_files():
    data = []
    for fname in os.listdir('../betty_number_sequences'):
        class_name = fname.split('_')[0]
        with open('../betty_number_sequences/'+fname, 'r') as f:
            bns = f.readlines()
        bns = ';'.join(map(str.strip, bns))
        data.append([bns, class_name])
    return data

def tokenize(data):
    M = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,',':10,';':11}
    C = {'cube':0,'cuboid':1,'cylinder':2,'ellipsoid':3,'line':4,'sphere':5,'torus':6}
    data = list(map(lambda dp: [list(map(M.get, dp[0])), C[dp[1]]], data))
    return data

def batchify(data: Tensor, labels: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    num_batches = data.size(0) // bsz
    seq_len = data.size(1)
    data = data[:num_batches * bsz]
    labels = labels[:num_batches * bsz]
    #print(data[0])
    data = data.view(num_batches, bsz, seq_len)#.t() #.contiguous()

    labels = labels.view(num_batches, bsz)
    return data, labels

data = load_data_from_files()
data = tokenize(data)
shuffle(data)
data_len = len(data)
val_data_len = int(data_len * 0.15)
test_data_len = val_data_len
val_data, data = data[:val_data_len], data[val_data_len:]
test_data, data = data[:test_data_len], data[test_data_len:]
train_data = data
train_data_len = len(train_data)

train_data, train_labels = map(list, zip(*train_data))
val_data, val_labels = map(list, zip(*val_data))
test_data, test_labels = map(list, zip(*test_data))


def pad_data(data, maxlen):
    #maxlen = max(map(len, data))
    for i in range(len(data)):
        seq = data[i]
        seq += [12] * (maxlen - len(seq))
        data[i] = seq

max_len = max([max(map(len, d)) for d in [train_data, val_data, test_data]])

pad_data(train_data, max_len)
pad_data(val_data, max_len)
pad_data(test_data, max_len)

train_data = torch.tensor(train_data)
train_labels = torch.tensor(train_labels)
val_data = torch.tensor(val_data)
val_labels = torch.tensor(val_labels)
test_data = torch.tensor(test_data)
test_labels = torch.tensor(test_labels)

#print(train_data[:5])

batch_size = 16
eval_batch_size = 8
train_data, train_labels = batchify(train_data, train_labels, batch_size)
val_data,val_labels = batchify(val_data, val_labels, eval_batch_size)
test_data, test_labels = batchify(test_data, test_labels, eval_batch_size)

#print(train_data[0])

ntokens = 13  # size of vocabulary
num_classes = 7
emsize = 100  # embedding dimension
d_hid = 100  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, num_classes, emsize, nhead, d_hid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 5
    start_time = time.time()
    epoch = 1
    seq_len = len(train_data[0][0])
    #print('sey_len', seq_len)
    src_mask = generate_square_subsequent_mask(seq_len).to(device)

    num_batches = len(train_data)
    #print(num_batches)
    for i in range(num_batches):
        data, targets = train_data[i], train_labels[i]
        #seq_len = data.size(0)
        #if seq_len != bptt:  # only on last batch
        #    src_mask = src_mask[:seq_len, :seq_len]
        #print(data)
        #print(targets)
        output = model(data.t(), src_mask)
        #print(output.shape)
        #print(output[-1])
        #loss = criterion(output.view(-1, num_classes), targets)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

for i in range(4):
    train(model)

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    seq_len = len(eval_data[0][0])
    src_mask = generate_square_subsequent_mask(seq_len).to(device)
    num_batches = len(eval_data)
    with torch.no_grad():
        for i in range(num_batches):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def score(model: nn.Module, eval_data: Tensor, eval_labels: Tensor):
    model.eval()
    seq_len = len(eval_data[0][0])
    src_mask = generate_square_subsequent_mask(seq_len).to(device)
    num_batches = len(eval_data)
    tp = 0
    tn = 0
    with torch.no_grad():
        for i in range(num_batches):
            data, targets = eval_data[i], eval_labels[i]
            output = model(data.t(), src_mask)
            output = output[-1]
            # output has shape [batch_size, num_classes]
            for j in range(len(output)):
                pred = int(output[j].argmax())
                if pred == targets[j]:
                    tp += 1
                else:
                    tn += 1
    print('tp', tp)
    print('tn', tn)
    print('acc', tp / (tp+tn))

score(model, train_data, train_labels)