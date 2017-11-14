import torch
from torch import cuda
from torch.autograd import Variable
import torch.nn as nn

torch.manual_seed(500)

class NMT(nn.Module):
    """docstring for NMT"""
    def __init__(self, src_vocab_size, trg_vocab_size, use_cuda=False):
        super(NMT, self).__init__()

        self.use_cuda = use_cuda

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.encoder_embed_size = 10
        self.decoder_embed_size = 10
        self.encoder_hidden_size = 10
        self.decoder_hidden_size = 2 * self.encoder_hidden_size

        self.encoderLSTM = nn.LSTM(self.encoder_embed_size, self.encoder_hidden_size, bidirectional=True)

        # Initialize decoder with weights parameters from original model
        self.decoder = nn.LSTMCell(self.decoder_embed_size + self.decoder_hidden_size, self.decoder_hidden_size)

        # Initialize embeddings
        self.encoder_embedding = nn.Embedding(src_vocab_size, self.encoder_embed_size)

        self.decoder_embedding = nn.Embedding(trg_vocab_size, self.decoder_embed_size)

        # Initialize Ws
        self.wi = nn.Linear(2*self.encoder_hidden_size,self.decoder_hidden_size, bias=False)

        self.wo = nn.Linear(2*self.decoder_hidden_size, self.decoder_hidden_size, bias=False)

        self.generator = nn.Linear(self.decoder_hidden_size, trg_vocab_size)

        self.softmax = torch.nn.Softmax()
        self.logsoftmax = torch.nn.LogSoftmax()
        self.tanh = torch.nn.Tanh()

        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.encoder_embedding = self.encoder_embedding.cuda()
            self.decoder_embedding = self.decoder_embedding.cuda()
            self.wi = self.wi.cuda()
            self.wo = self.wo.cuda()
            self.generator = self.generator.cuda()

    def forward(self, train_src_batch, train_trg_batch):

        encoder_batch, h, c = self.encoder(train_src_batch)
        sent_len = train_trg_batch.size()[0]
        batch_size = encoder_batch.size()[1]

        result = Variable(torch.FloatTensor(sent_len, batch_size, self.trg_vocab_size))
        # result[0] = Variable(torch.FloatTensor(batch_size, self.trg_vocab_size).fill_(-1))
        # result[0,:,2]=0
        w = Variable(torch.LongTensor(batch_size).fill_(2))

        if self.use_cuda:
            result = result.cuda()
            w = w.cuda()

        result[0] = Variable(torch.FloatTensor(batch_size, self.trg_vocab_size).fill_(-1))
        result[0,:,2]=0

        for i in range(1, sent_len):
            if self.training:

                result[i], h, c = self.decoderStep(encoder_batch, train_trg_batch[i-1], h, c)
            else:
                result[i], h, c = self.decoderStep(encoder_batch, w, h, c)
                _, w = torch.max(result[i], dim=1)
        return result


    def encoder(self, train_src_batch):
        encoder_input = self.encoder_embedding(train_src_batch)

        sys_out_batch, (h,c) = self.encoderLSTM(encoder_input)

        batch_size = sys_out_batch.size()[1]

        h = h.permute(1,2,0).contiguous().view(batch_size, 2*self.encoder_hidden_size)
        c = c.permute(1,2,0).contiguous().view(batch_size, 2*self.encoder_hidden_size)

        return sys_out_batch, h, c

    def decoderStep(self, sys_out_batch, w, h, c):

        seq_len = sys_out_batch.size()[0]
        batch_size = sys_out_batch.size()[1]

        wht1 = self.wi(h).view(1, batch_size, 2*self.encoder_hidden_size).expand_as(sys_out_batch)
        score = torch.sum(sys_out_batch * wht1, dim=2)

        score = torch.t(self.softmax(torch.t(score)))
            
        score = score.contiguous().view(seq_len,batch_size,1)

        st = torch.sum(score * sys_out_batch, dim=0)

        ct = self.tanh(self.wo(torch.cat([st, h], dim=1)))

        input = torch.cat([ct, self.decoder_embedding(w)], dim=1)

        h, c = self.decoder(input,(h,c))

        w = self.logsoftmax(self.generator(h))
        return w, h, c

