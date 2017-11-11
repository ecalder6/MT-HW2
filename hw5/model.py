import torch
from torch import cuda
from torch.autograd import Variable
import torch.nn as nn

class NMT(nn.Module):
    """docstring for NMT"""
    def __init__(self, original_model):
        super(NMT, self).__init__()

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
        self.decoder = nn.LSTM(1324, 1024)

        self.decoder.weight_ih_l0 = nn.Parameter(original_model['decoder.rnn.layers.0.weight_ih'])
        self.decoder.weight_hh_l0 = nn.Parameter(original_model['decoder.rnn.layers.0.weight_hh'])
        self.decoder.bias_ih_l0 = nn.Parameter(original_model['decoder.rnn.layers.0.bias_ih'])
        self.decoder.bias_hh_l0 = nn.Parameter(original_model['decoder.rnn.layers.0.bias_hh'])

        # Initialize embeddings
        self.encoder_embedding = nn.Embedding(36616, 300)
        self.decoder_embedding = nn.Embedding(23262, 300)
        self.encoder_embedding.weight = nn.Parameter(original_model['encoder.embeddings.emb_luts.0.weight'])
        print(original_model['encoder.embeddings.emb_luts.0.weight'].size())
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
        self.tanh = torch.nn.Tanh()

    def forward(self, train_trg_batch, sent_len):


        encoder_input = self.encoder_embedding(train_trg_batch)
        sys_out_batch, (encoder_hidden_states, _) = self.encoder(encoder_input)

        h = Variable(torch.FloatTensor(sys_out_batch.size()[1], 1024).fill_(1./1024))
        c = Variable(torch.FloatTensor(sys_out_batch.size()[1], 1024).fill_(0))

        w = self.softmax(self.generator(h))

        result = Variable(torch.FloatTensor(sent_len, sys_out_batch.size()[1], 23262))
        for i in range(sent_len):
            wht1 = self.wi(h).view(1, -1, 1024).expand_as(sys_out_batch)

            score = self.softmax(torch.sum(sys_out_batch * wht1, dim=2)).view(sys_out_batch.size()[0],sys_out_batch.size()[1],1)

            st = torch.sum(score * sys_out_batch, dim=0)
            ct = self.tanh(self.wo(torch.cat([st, h], dim=1)))

            _, w = torch.max(w, dim=1)
            input = torch.cat([self.decoder_embedding(w), ct], dim=1)
            input = input.view(1, input.size()[0], input.size()[1])

            _,(b,c) = self.decoder(input, (h,c))
            h = b[0]
            c = c[0]

            w = self.softmax(self.generator(h))
            result[i] = w

        return result


