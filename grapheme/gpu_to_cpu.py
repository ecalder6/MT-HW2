from __future__ import print_function
import utils.tensor
import utils.rand

import argparse
import dill
import logging

import sys

import torch
from torch import cuda
from torch.autograd import Variable
import torch.nn as nn

from model import NMT

class EnsembleNMT(nn.Module):
    """docstring for EnsembleNMT"""
    def __init__(self, models):
        super(EnsembleNMT, self).__init__()
        self.models = models
        self.num_models = len(models)

    def forward(self, train_src_batch, train_trg_batch):
        encoded_outputs, hidden = self.encoder(train_src_batch)

        sent_len = train_trg_batch.size()[0]
        batch_size = train_trg_batch.size()[1]

        results = Variable(torch.LongTensor(sent_len, batch_size))
        w = Variable(torch.LongTensor(batch_size).fill_(2))
        results[0] = w

        for i in range(1, sent_len-1):
            w, hidden = self.decoderStep(encoded_outputs, w, hidden)
            total_w = 0
            for w_prob in w:
                total_w += w_prob
            total_w[:,3] = torch.min(total_w).data[0]
            _, w = torch.max(total_w, dim=1) 

            results[i] = w   

        w = Variable(torch.LongTensor(batch_size).fill_(3))
        results[sent_len-1] = w
        return results

    def encoder(self, train_src_batch):
        encoded_outputs = []
        hidden = []

        for model in self.models:
            sys_out_batch, h, c = model.encoder(train_src_batch)
            encoded_outputs.append(sys_out_batch)
            hidden.append((h,c))

        return encoded_outputs, hidden


    def decoderStep(self, sys_out_batch, w, hidden):
        total_w = []
        next_hidden = []
        for i in range(self.num_models):
            # sys_out_batch[i]
            h,c = hidden[i]

            # print(sys_out_batch[i].size())
            # print(w.size())
            # print(h.size())
            # print(c.size())

            w_2,h,c = self.models[i].decoderStep(sys_out_batch[i], w, h, c)
            next_hidden.append((h,c))
            total_w.append(w_2)

        return total_w, next_hidden



filenames = [ \
'model_save.nll_0.73.epoch_15', \
'model_save.nll_0.75.epoch_16', \
'model_save.nll_0.73.epoch_17', \
'model_save.nll_0.71.epoch_18', \
'model_save.nll_0.71.epoch_19' \
]

models = []
for filename in filenames:
    model = torch.load(filename, map_location={'cuda:0':'cpu'})
    model.use_cuda = False
    model.eval()
    models.append(model)

enmt = EnsembleNMT(models)
enmt.eval()

# print("Load")
_, _, src_test, src_vocab = torch.load(open("data/hw5.words", 'rb'))
# print("Load src")
_, _, trg_test, trg_vocab = torch.load(open("data/hw5.phoneme", 'rb'))
# print("Load trg")
# sys.exit(0)

# batched_test_src, batched_test_src_mask, sort_index = utils.tensor.advanced_batchize(src_test, 1, src_vocab.stoi["<blank>"])
# batched_test_trg, batched_test_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_test, 1, trg_vocab.stoi["<blank>"], sort_index)

trg_vocab_size = len(trg_vocab)
src_vocab_size = len(src_vocab)
dev_loss = 0
criterion = torch.nn.NLLLoss()
# print(len(batched_test_src))
# for i, batch_i in enumerate(utils.rand.srange(len(batched_test_src))):
for batch_i in range(len(src_test)):
  print("{0}/ {1}".format(batch_i, len(src_test) ), file=sys.stderr)
  test_src_batch = Variable(src_test[batch_i], volatile=True)  # of size (src_seq_len, batch_size)
  test_trg_batch = Variable(trg_test[batch_i], volatile=True)  # of size (src_seq_len, batch_size)
  # test_src_mask = Variable(batched_test_src_mask[batch_i], volatile=True)
  # test_trg_mask = Variable(batched_test_trg_mask[batch_i], volatile=True)
  # print(test_src_batch.size())
  test_src_batch = test_src_batch.view(-1, 1)
  test_trg_batch = test_trg_batch.view(-1, 1)
  # print(test_trg_batch.size())
  sys_out_batch = enmt(test_src_batch, test_trg_batch)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary 

  for j in range(sys_out_batch.size()[1]):
    sent = []
    for i in range(1, sys_out_batch.size()[0]):
      # print(sys_out_batch[i,j].data.numpy()[0])
      sent.append(trg_vocab.itos[sys_out_batch[i,j].data.numpy()[0]])
    print(' '.join(sent).encode('utf-8').strip())

  # for j in range(sys_out_batch.size()[1]):
  #   sent = []
  #   for i in range(1, sys_out_batch.size()[0]):
  #     # print(sys_out_batch[i,j].data.numpy()[0])
  #     sent.append(trg_vocab.itos[test_trg_batch[i,j].data.numpy()[0]])
  #   print(' '.join(sent).encode('utf-8').strip())


  # if i % 1000 == 0:
    # logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))

  # model.eval()
  # sys_out_batch = model(test_src_batch, test_trg_batch)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary 
  # test_trg_mask = test_trg_mask.view(-1)
  # test_trg_batch = test_trg_batch.view(-1)
  # test_trg_batch = test_trg_batch.masked_select(test_trg_mask)
  # test_trg_mask = test_trg_mask.unsqueeze(1).expand(len(test_trg_mask), trg_vocab_size)
  # sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
  # sys_out_batch = sys_out_batch.masked_select(test_trg_mask).view(-1, trg_vocab_size)
  # loss = criterion(sys_out_batch, test_trg_batch)
  # dev_loss += loss

# dev_avg_loss = dev_loss / len(batched_test_src)
# print(dev_avg_loss)
# logging.info("Average loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss.data[0], epoch_i))




# 