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

import numpy as np

from model import NMT

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW5.")
parser.add_argument("--data_file", required=True,
                    help="File prefix for training set.")
parser.add_argument("--src_lang", default="de",
                    help="Source Language. (default = de)")
parser.add_argument("--trg_lang", default="en",
                    help="Target Language. (default = en)")
parser.add_argument("--original_model_file", required=True,
                    help="Location to load the original model.")
parser.add_argument("--model_file", required=True,
                    help="Location to dump the models.")
parser.add_argument("--batch_size", default=4, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=SGD)")
parser.add_argument("--learning_rate", "-lr", default=0.1, type=float,
                    help="Learning rate of the optimization. (default=0.1)")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum when performing SGD. (default=0.9)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need

def main(options):

  original_model = torch.load(open(options.original_model_file, 'rb'))

  nmt = NMT(original_model)

  # use_cuda = (len(options.gpuid) >= 1)
  # if options.gpuid:
  #   cuda.set_device(options.gpuid[0])

  src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  batched_test_src, batched_test_src_mask, _ = utils.tensor.advanced_batchize(src_test, 24, src_vocab.stoi["<pad>"])
  batched_test_trg, batched_test_trg_mask, _ = utils.tensor.advanced_batchize(trg_test, 24, trg_vocab.stoi["<pad>"])

  # total_loss = 0
  # total_sent = 0

  # nmt(Variable(torch.from_numpy(np.array([[46, 68], [470, 72], [30, 4]]))),Variable(torch.from_numpy(np.array([[1],[1]]))))
  # sys.exit(0)
  # print(torch.min(src_test))
  # print(torch.max(trg_test))
  for i, batch_i in enumerate(utils.rand.srange(len(batched_test_src))):
      print(i)
      test_src_batch = Variable(batched_test_src[batch_i], volatile=True)  # of size (src_seq_len, batch_size)
      test_trg_batch = Variable(batched_test_trg[batch_i], volatile=True)  # of size (src_seq_len, batch_size)
      test_src_mask = Variable(batched_test_src_mask[batch_i], volatile=True)
      test_trg_mask = Variable(batched_test_trg_mask[batch_i], volatile=True)

      total_sent += test_src_batch.size()[0]

      if use_cuda:
        test_src_batch = test_src_batch.cuda()
        test_trg_batch = test_trg_batch.cuda()
        test_src_mask = test_src_mask.cuda()
        test_trg_mask = test_trg_mask.cuda()
      # print(torch.min(test_src_batch))
      # print(torch.max(test_src_batch))
      # print(test_src_batch)

      sys_out_batch = nmt(test_src_batch, test_trg_batch.size()[0])
      test_trg_mask = test_trg_mask.view(-1)
      test_trg_batch = test_trg_batch.view(-1)
      test_trg_batch = test_trg_batch.masked_select(test_trg_mask)
      test_trg_mask = test_trg_mask.unsqueeze(1).expand(len(test_trg_mask), trg_vocab_size - 1)

      sys_out_batch = sys_out_batch.view(-1, trg_vocab_size - 1)
      sys_out_batch = sys_out_batch.masked_select(test_trg_mask).view(-1, trg_vocab_size - 1)

      loss = criterion(sys_out_batch, test_trg_batch)
      logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))

      total_loss += loss
      # break
      # _, sys_out_batch = torch.max(sys_out_batch, dim=2)
      # sys_out_batch = sys_out_batch.view(-1)
      # sent = []
      # # print(sys_out_batch)
      # for w in sys_out_batch:
      #   # print(w)
      #   sent.append(trg_vocab.itos[w.data[0]])
      # # print(sent)
      # # print(sent.join())
      # print(' '.join(sent).encode('utf-8').strip())
  print(total_loss, total_sent)
  print(total_loss/total_sent)
  print(torch.exp(total_loss/total_sent))
  sys.exit(0)

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
