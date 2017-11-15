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
from model import NMT

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW5.")
parser.add_argument("--data_file", default="data/hw5",
                    help="File prefix for training set.")
parser.add_argument("--src_lang", default="words",
                    help="Source Language. (default = words)")
parser.add_argument("--trg_lang", default="phoneme",
                    help="Target Language. (default = phoneme)")
parser.add_argument("--model_file", required=True,
                    help="Location to load the models.")
parser.add_argument("--batch_size", default=24, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.1)")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum when performing SGD. (default=0.9)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need

def main(options):

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

  _, _, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  _, _, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  batched_test_src, batched_test_src_mask, sort_index = utils.tensor.advanced_batchize(src_test, options.batch_size, src_vocab.stoi["<blank>"])
  batched_test_trg, batched_test_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_test, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)

  trg_vocab_size = len(trg_vocab)
  # print(trg_vocab.itos[4])

  nmt = torch.load(options.model_file, map_location={'cuda:0': 'cpu'})
  nmt.use_cuda = False
  nmt.eval()
  if use_cuda > 0:
    nmt.cuda()
  else:
    nmt.cpu()

  criterion = torch.nn.NLLLoss()
  # optimizer = eval("torch.optim." + options.optimizer)(nmt.parameters(), options.learning_rate)

  total_loss = 0
  num_sents = 0
  for i, batch_i in enumerate(utils.rand.srange(len(batched_test_src))):
    print("{0}/ {1}".format(i, len(batched_test_src) ), file=sys.stderr)
    test_src_batch = Variable(batched_test_src[batch_i])  # of size (src_seq_len, batch_size)
    test_trg_batch = Variable(batched_test_trg[batch_i])  # of size (src_seq_len, batch_size)
    test_src_mask = Variable(batched_test_src_mask[batch_i])
    test_trg_mask = Variable(batched_test_trg_mask[batch_i])
    if use_cuda:
      test_src_batch = test_src_batch.cuda()
      test_trg_batch = test_trg_batch.cuda()
      test_src_mask = test_src_mask.cuda()
      test_trg_mask = test_trg_mask.cuda()
    num_sents += 1

    sys_out_batch = nmt(test_src_batch, test_trg_batch)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary 
    _, max = torch.max(sys_out_batch, dim=2) # (trg_seq_len, batch_size)

    for j in range(max.size()[1]):
      sent = []
      for i in range(1, max.size()[0]):
        sent.append(trg_vocab.itos[max[i,j].data.numpy()[0]])
      try:
        sent = sent[:sent.index('</s>') + 1]
      except ValueError:
        pass
      print(' '.join(sent).encode('utf-8').strip())

    # test_trg_mask = test_trg_mask.view(-1)
    # test_trg_batch = test_trg_batch.view(-1)
    # test_trg_batch = test_trg_batch.masked_select(test_trg_mask)
    # test_trg_mask = test_trg_mask.unsqueeze(1).expand(len(test_trg_mask), trg_vocab_size)
    # sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
    # sys_out_batch = sys_out_batch.masked_select(test_trg_mask).view(-1, trg_vocab_size)
    # loss = criterion(sys_out_batch, test_trg_batch)
    # _, max = torch.max(sys_out_batch,dim=1)
    # print(sys_out_batch[dev_trg_batch])
    # print(max, dev_trg_batch)
    # total_loss += loss
    # break
    # print(total_loss, num_sents)
    # print(total_loss/num_sents)
    # print(torch.exp(total_loss/num_sents))


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
