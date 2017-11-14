import torch
from torch import cuda
from torch.autograd import Variable
import torch.nn as nn

torch.manual_seed(500)

class NMT(nn.Module):
    """docstring for NMT"""
    def __init__(self, original_model, use_cuda=False):
        super(NMT, self).__init__()
        
        self.use_cuda = use_cuda
        if use_cuda:
            for elem in original_model:
                original_model[elem] = original_model[elem].cuda()

        self.encoder = nn.LSTM(300, 512, bidirectional=True)

        self.encoder.weight_ih_l0 = nn.Parameter(original_model['encoder.rnn.weight_ih_l0'])
        self.encoder.weight_hh_l0 = nn.Parameter(original_model['encoder.rnn.weight_hh_l0'])
        self.encoder.bias_ih_l0 = nn.Parameter(original_model['encoder.rnn.bias_ih_l0'])
        self.encoder.bias_hh_l0 = nn.Parameter(original_model['encoder.rnn.bias_hh_l0'])

        self.encoder.weight_ih_l0_reverse = nn.Parameter(original_model['encoder.rnn.weight_ih_l0_reverse'])
        self.encoder.weight_hh_l0_reverse = nn.Parameter(original_model['encoder.rnn.weight_hh_l0_reverse'])
        self.encoder.bias_ih_l0_reverse = nn.Parameter(original_model['encoder.rnn.bias_ih_l0_reverse'])
        self.encoder.bias_hh_l0_reverse = nn.Parameter(original_model['encoder.rnn.bias_hh_l0_reverse'])

        # Initialize decoder with weights parameters from original model
        self.decoder = nn.LSTMCell(1324, 1024)

        self.decoder.weight_ih = nn.Parameter(original_model['decoder.rnn.layers.0.weight_ih'])
        self.decoder.weight_hh = nn.Parameter(original_model['decoder.rnn.layers.0.weight_hh'])
        self.decoder.bias_ih = nn.Parameter(original_model['decoder.rnn.layers.0.bias_ih'])
        self.decoder.bias_hh = nn.Parameter(original_model['decoder.rnn.layers.0.bias_hh'])

        # Initialize embeddings
        self.encoder_embedding = nn.Embedding(36616, 300)
        self.encoder_embedding.weight = nn.Parameter(original_model['encoder.embeddings.emb_luts.0.weight'])

        self.decoder_embedding = nn.Embedding(23262, 300)
        self.decoder_embedding.weight = nn.Parameter(original_model['decoder.embeddings.emb_luts.0.weight'])

        # Initialize Ws
        self.wi = nn.Linear(1024,1024, bias=False)
        self.wi.weight = nn.Parameter(original_model['decoder.attn.linear_in.weight'])

        self.wo = nn.Linear(2048, 1024, bias=False)
        self.wo.weight = nn.Parameter(original_model['decoder.attn.linear_out.weight'])

        self.generator = nn.Linear(1024, 23262)
        self.generator.weight = nn.Parameter(original_model['0.weight'])
        self.generator.bias = nn.Parameter(original_model['0.bias'])

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

        sent_len = train_trg_batch.size()[0]

        encoder_input = self.encoder_embedding(train_src_batch)

        sys_out_batch, (h,c) = self.encoder(encoder_input)

        seq_len = sys_out_batch.size()[0]
        batch_size = sys_out_batch.size()[1]

        h = h.permute(1,2,0).contiguous().view(batch_size, 1024)
        c = c.permute(1,2,0).contiguous().view(batch_size, 1024)

        encoder_hidden_size = 512
        decoder_hidden_size = 1024

        result = Variable(torch.FloatTensor(sent_len, batch_size, 23262))

        w = Variable(torch.LongTensor(batch_size).fill_(2))
        if self.use_cuda:
            result = result.cuda()
            w = w.cuda()

        result[0] = Variable(torch.FloatTensor(batch_size, 23262).fill_(0))

        for i in range(1, sent_len):
            wht1 = self.wi(h).view(1, batch_size, 2*encoder_hidden_size).expand_as(sys_out_batch)
            score = torch.sum(sys_out_batch * wht1, dim=2)

            score = torch.t(self.softmax(torch.t(score)))
                
            score = score.contiguous().view(seq_len,batch_size,1)

            st = torch.sum(score * sys_out_batch, dim=0)

            ct = self.tanh(self.wo(torch.cat([st, h], dim=1)))

            if self.training:
                input = torch.cat([ct, self.decoder_embedding(train_trg_batch[i-1])], dim=1)
            else:
                input = torch.cat([ct, self.decoder_embedding(w)], dim=1)

            h, c = self.decoder(input,(h,c))

            w = self.logsoftmax(self.generator(h))

            result[i] = w
            _, w = torch.max(w, dim=1)

        return result


