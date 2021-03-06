from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import cv2 as cv
import random

def logsumexp(value, dim=None, keepdim=True):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    return m + torch.log(torch.sum(torch.exp(value0),
                                   dim=dim, keepdim=True))

class RNN(nn.Module):

    def __init__(self, vocab_size, hidden_size = 16, embedding_size=32):
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embeddings = Variable(torch.randn(vocab_size, embedding_size), requires_grad=True)
        self.W_x = Variable(torch.randn(embedding_size, hidden_size), requires_grad=True)
        self.b_x = Variable(torch.randn(hidden_size), requires_grad=True)
        self.W_h = Variable(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.b_h = Variable(torch.randn(hidden_size), requires_grad=True)
        self.output = Variable(torch.randn(hidden_size, vocab_size), requires_grad=True)

    def forward(self, x):
        encode = self.embeddings[x.data,:]
        seq_length = x.size()[0]
        batch_size = x.size()[1]
        h = self.init_hidden(batch_size)
        total_h = Variable(torch.FloatTensor(seq_length, batch_size, self.hidden_size))
        for t, step in enumerate(encode):
            print(t)
            a = step.matmul(self.W_x) + self.b_x
            b = h.matmul(self.W_h) + self.b_h
            c = a + b
            h = self.sigmoid(c)

            total_h[t] = h

        a = total_h.matmul(self.output)
        return self.logsoftmax(a)

    def logsoftmax(self, a):
        return a - logsumexp(a, 2).expand_as(a)

    def sigmoid(self, c):
        return 1. / (1. + c.mul(-1).exp())

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))

class BiRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size = 8, embedding_size=32):
        super(BiRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embeddings = Variable(torch.randn(vocab_size, embedding_size), requires_grad=True)

        self.W_x1 = Variable(torch.randn(embedding_size, hidden_size), requires_grad=True)
        self.b_x1 = Variable(torch.randn(hidden_size), requires_grad=True)
        self.W_h1 = Variable(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.b_h1 = Variable(torch.randn(hidden_size), requires_grad=True)

        self.W_x2 = Variable(torch.randn(embedding_size, hidden_size), requires_grad=True)
        self.b_x2 = Variable(torch.randn(hidden_size), requires_grad=True)
        self.W_h2 = Variable(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.b_h2 = Variable(torch.randn(hidden_size), requires_grad=True)

        self.output = Variable(torch.randn(2*hidden_size, vocab_size), requires_grad=True)

    def forward(self, x):
        encode = self.embeddings[x.data,:]
        seq_length = x.size()[0]
        batch_size = x.size()[1]
        h = self.init_hidden(batch_size)
        total_h1 = Variable(torch.FloatTensor(seq_length, batch_size, self.hidden_size))

        for t, step in enumerate(encode):
            total_h1[t] = h
            print(t)
            if t == seq_length - 1:
                break
            a = step.matmul(self.W_x1) + self.b_x1
            b = h.matmul(self.W_h1) + self.b_h1
            c = a + b
            h = self.sigmoid(c)

        h = self.init_hidden(batch_size)
        total_h2 = Variable(torch.FloatTensor(seq_length, batch_size, self.hidden_size))
        for t, step in enumerate(reversed(encode)):
            print(seq_length-t-1)
            total_h2[t] = h
            if t == seq_length - 1:
                break
            a = step.matmul(self.W_x2) + self.b_x2
            b = h.matmul(self.W_h2) + self.b_x2
            c = a + b
            h = self.sigmoid(c)

        total_h = torch.cat((total_h1, total_h2), 2)
        a = total_h.matmul(self.output)
        return self.logsoftmax(a)

    def logsoftmax(self, a):
        return a - logsumexp(a, 2).expand_as(a)

    def sigmoid(self, c):
        return 1. / (1. + c.mul(-1).exp())

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))

class BiGRU(nn.Module):

    def __init__(self, vocab_size, hidden_size = 8, embedding_size=32, dropout=None):
        super(BiGRU, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout

        self.embeddings = Variable(torch.randn(vocab_size, embedding_size), requires_grad=True)
        self.W_z1 = nn.Linear(embedding_size + hidden_size, 1)
        self.W_r1 = nn.Linear(embedding_size + hidden_size, 1)
        self.W_h1 = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.W_z2 = nn.Linear(embedding_size + hidden_size, 1)
        self.W_r2 = nn.Linear(embedding_size + hidden_size, 1)
        self.W_h2 = nn.Linear(embedding_size + hidden_size, hidden_size)

        self.output = nn.Linear(2*hidden_size, vocab_size)

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.LogSoftmax()
        self.tanh = nn.Tanh()

    def forward(self, x):
        encode = self.embeddings[x.data,:]
        seq_length = x.size()[0]
        batch_size = x.size()[1]
        h = self.init_hidden(batch_size)
        total_h1 = Variable(torch.FloatTensor(seq_length, batch_size, self.hidden_size))

        for t, step in enumerate(encode):
            total_h1[t] = h
            print(t)

            if self.dropout and self.training:
                step_mask = Variable(torch.bernoulli(
                    torch.Tensor(batch_size, self.embedding_size).fill_(1. - self.dropout)), requires_grad=False) / self.dropout
                h_mask = Variable(torch.bernoulli(
                    torch.Tensor(batch_size, self.hidden_size).fill_(1. - self.dropout)), requires_grad=False) / self.dropout
                step = step * step_mask
                h = h * h_mask

            if t == seq_length - 1:
                break
            
            z_t = self.sigmoid(self.W_z1(torch.cat((h, step),1))).expand_as(h)
            r_t = self.sigmoid(self.W_r1(torch.cat((h, step),1))).expand_as(h)
            h_t1 = self.tanh(self.W_h1(torch.cat((r_t*h, step), 1)))
            h = (1. - z_t) * h + z_t * h_t1

        h = self.init_hidden(batch_size)
        total_h2 = Variable(torch.FloatTensor(seq_length, batch_size, self.hidden_size))
        for t, step in enumerate(reversed(encode)):
            print(seq_length-t-1)
            total_h2[t] = h
            if t == seq_length - 1:
                break

            z_t = self.sigmoid(self.W_z2(torch.cat((h, step),1))).expand_as(h)
            r_t = self.sigmoid(self.W_r2(torch.cat((h, step),1))).expand_as(h)
            h_t2 = self.tanh(self.W_h2(torch.cat((r_t*h, step), 1)))
            h = (1. - z_t) * h + z_t * h_t2

        total_h = torch.cat((total_h1, total_h2), 2)
        a = self.output(total_h)
        return self.logsoftmax(a)

    def logsoftmax(self, a):
        return a - logsumexp(a, 2).expand_as(a)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))


model = BiRNN(5, 4, 8)
model.train()
word_idx = Variable(torch.LongTensor([[0, 3], [1, 3], [2, 3]]))
model(word_idx)
