import torch
from torch import cuda
from torch.autograd import Variable
import torch.nn as nn

class LM(nn.Module):
    """docstring for NMT"""
    def __init__(self, vocab_size, bos, eos, embed_size, hidden_size, use_cuda=False):
        super(LM, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.vocab_size = vocab_size
        self.bos = bos
        self.eos = eos
        self.use_cuda = use_cuda

        self.encoder = nn.LSTM(self.embed_size, self.hidden_size)
        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.generator = nn.Linear(self.hidden_size, vocab_size)

        self.logsoftmax = torch.nn.LogSoftmax()

        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.embedding = self.embedding.cuda()
            self.generator = self.generator.cuda()


    def forward(self, sent=None, h=None, c=None, encode=True, teacher_forcing=False, tgt_sent=None):

        if not encode and (h is None or c is None):
            raise ValueError('When decoding, h and c must be given')

        if h is None or c is None:
            if sent is not None:
                batch_size = sent.size()[1]
            elif tgt_sent is not None:
                batch_size = tgt_sent.size()[1]
            else:
                raise ValueError('Either need h and c or batch size')

        if h is None:
            h = self.init_hidden(batch_size)
        if c is None:
            c = self.init_hidden(batch_size)

        if encode:
            x = self.embedding(sent)
            _, (h,c) = self.encoder(x)
            return h,c
        else:
            if tgt_sent is None:
                raise ValueError()
                # batch_size = x.size()[1]
                # if batch_size != 1:
                #     raise ValueError('Batch Size must be only 1')

                # result = []
                # result.append(torch.FloatTensor(batch_size, self.vocab_size).fill_(-1))
                # result[0][:, self.bos] = 0
                # w = Variable(torch.LongTensor(batch_size).fill_(self.bos))

                # i = 0
                # while w[0] != self.eos:
                #     i += 1
                #     _, (h,c) = self.encoder(self.embedding(w), (h,c))
                #     result.append(self.generator(h))
                #     _, w = torch.max(result[i], dim=1)

                # var_result = Variable(torch.FloatTensor(len(result), batch_size, self.vocab_size))
                # for i, res in enumerate(result):
                #     var_result[i] = res
                # return var_result
            else:
                sent_len = tgt_sent.size()[0]
                batch_size = tgt_sent.size()[1]

                result = Variable(torch.FloatTensor(sent_len, batch_size, self.vocab_size))
                if self.use_cuda:
                    result = result.cuda()
                result[0] = Variable(torch.FloatTensor(batch_size, self.vocab_size).fill_(-1))
                result[0,:,self.bos] = 0

                if not teacher_forcing:
                    w = Variable(torch.LongTensor(batch_size).fill_(self.bos))
                    if self.use_cuda():
                        w = w.cuda()

                for i in range(1, sent_len):
                    # print(i)
                    if teacher_forcing:
                        # print(tgt_sent[i-1])
                        # print(self.embedding(tgt_sent[i-1]))
                        _, (h,c) = self.encoder(self.embedding(tgt_sent[i-1]).view(1,batch_size,-1), (h,c))
                    else:
                        _, (h,c) = self.encoder(self.embedding(w).view(1,batch_size,-1), (h,c))

                    result[i] = self.logsoftmax(self.generator(h).view(batch_size, -1))
                    if not teacher_forcing:
                        _, w = torch.max(result[i], dim=1)

                return result

    def init_hidden(self, batch_size):
        if self.use_cuda:
            return Variable(torch.FloatTensor(1, batch_size, self.hidden_size).fill_(0)).cuda()

        return Variable(torch.FloatTensor(1, batch_size, self.hidden_size).fill_(0))
