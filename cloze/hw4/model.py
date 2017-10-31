import torch
import torch.utils.data
import torch.nn as nn
import torch.cunn as cunn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

def logsumexp(value, dim=None, keepdim=True):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    return m + torch.log(torch.sum(torch.exp(value0),
                                   dim=dim, keepdim=True))

class RNNLM(nn.Module):

    def __init__(self, vocab_size, hidden_size = 16, embedding_size=32,
            use_cuda=False):
        super(RNNLM, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.use_cuda = use_cuda

        if use_cuda:
            self.embeddings = cunn.Parameter(torch.randn(vocab_size, embedding_size), requires_grad=True)
            self.W_x = cunn.Parameter(torch.randn(embedding_size, hidden_size), requires_grad=True)
            self.b_x = cunn.Parameter(torch.randn(hidden_size), requires_grad=True)
            self.W_h = cunn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
            self.b_h = cunn.Parameter(torch.randn(hidden_size), requires_grad=True)
            self.output = cunn.Parameter(torch.randn(hidden_size, vocab_size), requires_grad=True)

        else:
            self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_size), requires_grad=True)
            self.W_x = nn.Parameter(torch.randn(embedding_size, hidden_size), requires_grad=True)
            self.b_x = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
            self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
            self.b_h = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
            self.output = nn.Parameter(torch.randn(hidden_size, vocab_size), requires_grad=True)

    def forward(self, x):
        encode = self.embeddings[x.data,:]
        seq_length = x.size()[0]
        batch_size = x.size()[1]
        h = self.init_hidden(batch_size)
        total_h = Variable(torch.FloatTensor(seq_length, batch_size, self.hidden_size))
        if self.use_cuda:
            total_h = Variable(torch.FloatTensor(seq_length, batch_size,
                self.hidden_size)).cuda()

        
        for t, step in enumerate(encode):
            # print(t)
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

class BiRNNLM(nn.Module):

    def __init__(self, vocab_size, hidden_size = 8, embedding_size=32):
        super(BiRNNLM, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_size), requires_grad=True)

        self.W_x1 = nn.Parameter(torch.randn(embedding_size, hidden_size), requires_grad=True)
        self.b_x1 = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        self.W_h1 = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.b_h1 = nn.Parameter(torch.randn(hidden_size), requires_grad=True)

        self.W_x2 = nn.Parameter(torch.randn(embedding_size, hidden_size), requires_grad=True)
        self.b_x2 = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        self.W_h2 = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.b_h2 = nn.Parameter(torch.randn(hidden_size), requires_grad=True)

        self.output = nn.Parameter(torch.randn(2*hidden_size, vocab_size), requires_grad=True)

    def forward(self, x):
        encode = self.embeddings[x.data,:]
        seq_length = x.size()[0]
        batch_size = x.size()[1]
        h = self.init_hidden(batch_size)
        total_h1 = Variable(torch.FloatTensor(seq_length, batch_size, self.hidden_size))

        for t, step in enumerate(encode):
            total_h1[t] = h
            #print(t)
            if t == seq_length - 1:
                break
            a = step.matmul(self.W_x1) + self.b_x1
            b = h.matmul(self.W_h1) + self.b_h1
            c = a + b
            h = self.sigmoid(c)
            #total_h1[t] = h

        h = self.init_hidden(batch_size)
        total_h2 = Variable(torch.FloatTensor(seq_length, batch_size, self.hidden_size))
        for t, step in enumerate(reversed(encode)):
            # print(seq_length-t-1)
            total_h2[seq_length - t -1] = h
            if t == seq_length - 1:
                break
            a = step.matmul(self.W_x2) + self.b_x2
            b = h.matmul(self.W_h2) + self.b_h2
            c = a + b
            h = self.sigmoid(c)
            #total_h2[t] = h

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

    def __init__(self, vocab_size, hidden_size = 8, embedding_size=32, dropout=0.4):
        super(BiGRU, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout

        self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_size), requires_grad=True)
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
            # print(t)

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
            # print(seq_length-t-1)
            total_h2[seq_length-t-1] = h
            if self.dropout and self.training:
                step_mask = Variable(torch.bernoulli(
                    torch.Tensor(batch_size, self.embedding_size).fill_(1. - self.dropout)), requires_grad=False) / self.dropout
                h_mask = Variable(torch.bernoulli(
                    torch.Tensor(batch_size, self.hidden_size).fill_(1. - self.dropout)), requires_grad=False) / self.dropout
                step = step * step_mask
                h = h * h_mask
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

class BiLSTM(nn.Module):

    def __init__(self, vocab_size, hidden_size = 8, embedding_size=32, cell_size = 8, dropout=0.4):
        super(BiLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.cell_size = cell_size
        self.dropout = dropout

        self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_size), requires_grad=True)
        self.W_f1 = nn.Linear(embedding_size + hidden_size, 1)
        self.W_i1 = nn.Linear(embedding_size + hidden_size, 1)
        self.W_C1 = nn.Linear(embedding_size + hidden_size, cell_size)
        self.W_o1 = nn.Linear(embedding_size + hidden_size, 1)

        self.W_f2 = nn.Linear(embedding_size + hidden_size, 1)
        self.W_i2 = nn.Linear(embedding_size + hidden_size, 1)
        self.W_C2 = nn.Linear(embedding_size + hidden_size, cell_size)
        self.W_o2 = nn.Linear(embedding_size + hidden_size, 1)

        self.output = nn.Linear(2*hidden_size, vocab_size)

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.LogSoftmax()
        self.tanh = nn.Tanh()

    def forward(self, x):
        encode = self.embeddings[x.data,:]
        seq_length = x.size()[0]
        batch_size = x.size()[1]
        h = self.init_hidden(batch_size)
        C = self.init_cell(batch_size)
        total_h1 = Variable(torch.FloatTensor(seq_length, batch_size, self.hidden_size))

        for t, step in enumerate(encode):
            total_h1[t] = h
            # print(t)

            if self.dropout and self.training:
                step_mask = Variable(torch.bernoulli(
                    torch.Tensor(batch_size, self.embedding_size).fill_(1. - self.dropout)), requires_grad=False) / self.dropout
                h_mask = Variable(torch.bernoulli(
                    torch.Tensor(batch_size, self.hidden_size).fill_(1. - self.dropout)), requires_grad=False) / self.dropout
                C_mask = Variable(torch.bernoulli(
                    torch.Tensor(batch_size, self.cell_size).fill_(1. - self.dropout)), requires_grad=False) / self.dropout
                step = step * step_mask
                h = h * h_mask
                C = C * C_mask

            if t == seq_length - 1:
                break
            
            f_t = self.sigmoid(self.W_f1(torch.cat((h, step),1))).expand_as(C)
            i_t = self.sigmoid(self.W_i1(torch.cat((h, step),1))).expand_as(C)
            C_t1 = self.tanh(self.W_C1(torch.cat((h,step),1)))
            C = f_t * C + i_t + C_t1

            o_t = self.sigmoid(self.W_o1(torch.cat((h,step),1))).expand_as(C)
            h = o_t * self.tanh(C)

        h = self.init_hidden(batch_size)
        total_h2 = Variable(torch.FloatTensor(seq_length, batch_size, self.hidden_size))
        for t, step in enumerate(reversed(encode)):
            # print(seq_length-t-1)
            total_h2[seq_length-t-1] = h
            if self.dropout and self.training:
                step_mask = Variable(torch.bernoulli(
                    torch.Tensor(batch_size, self.embedding_size).fill_(1. - self.dropout)), requires_grad=False) / self.dropout
                h_mask = Variable(torch.bernoulli(
                    torch.Tensor(batch_size, self.hidden_size).fill_(1. - self.dropout)), requires_grad=False) / self.dropout
                C_mask = Variable(torch.bernoulli(
                    torch.Tensor(batch_size, self.cell_size).fill_(1. - self.dropout)), requires_grad=False) / self.dropout
                step = step * step_mask
                h = h * h_mask
                C = C * C_mask

            if t == seq_length - 1:
                break

            f_t = self.sigmoid(self.W_f2(torch.cat((h, step),1))).expand_as(C)
            i_t = self.sigmoid(self.W_i2(torch.cat((h, step),1))).expand_as(C)
            C_t1 = self.tanh(self.W_C2(torch.cat((h,step),1)))
            C = f_t * C + i_t + C_t1

            o_t = self.sigmoid(self.W_o2(torch.cat((h,step),1))).expand_as(C)
            h = o_t * self.tanh(C)

        total_h = torch.cat((total_h1, total_h2), 2)
        a = self.output(total_h)
        return self.logsoftmax(a)

    def logsoftmax(self, a):
        return a - logsumexp(a, 2).expand_as(a)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))

    def init_cell(self, batch_size):
        return Variable(torch.zeros(batch_size, self.cell_size))
