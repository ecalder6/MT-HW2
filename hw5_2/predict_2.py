import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
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
parser.add_argument("--model_file", required=True,
                    help="Location to dump the models.")
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
parser.add_argument("--original_model_file", required=True,
                    help="Location to load the original model.")
# feel free to add more arguments as you need

def main(options):

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

  _, src_dev, _, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  _, trg_dev, _, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  batched_dev_src, batched_dev_src_mask, sort_index = utils.tensor.advanced_batchize(src_dev, options.batch_size, src_vocab.stoi["<blank>"])
  batched_dev_trg, batched_dev_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_dev, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)

  trg_vocab_size = len(trg_vocab)
  print(trg_vocab.itos[4])

  original_model = torch.load(open(options.original_model_file, 'rb'))
  nmt = NMT(original_model) # TODO: add more arguments as necessary 
  nmt.eval()
  if use_cuda > 0:
    nmt.cuda()
  else:
    nmt.cpu()

  criterion = torch.nn.NLLLoss()
  # optimizer = eval("torch.optim." + options.optimizer)(nmt.parameters(), options.learning_rate)

  total_loss = 0
  num_sents = 0
  for i, batch_i in enumerate(utils.rand.srange(len(batched_dev_src))):
    print("{0}/ {1}".format(i, len(batched_dev_src)))
    dev_src_batch = Variable(batched_dev_src[batch_i])  # of size (src_seq_len, batch_size)
    dev_trg_batch = Variable(batched_dev_trg[batch_i])  # of size (src_seq_len, batch_size)
    dev_src_mask = Variable(batched_dev_src_mask[batch_i])
    dev_trg_mask = Variable(batched_dev_trg_mask[batch_i])
    if use_cuda:
      dev_src_batch = dev_src_batch.cuda()
      dev_trg_batch = dev_trg_batch.cuda()
      dev_src_mask = dev_src_mask.cuda()
      dev_trg_mask = dev_trg_mask.cuda()
    num_sents += 1

    sys_out_batch = nmt(dev_src_batch, dev_trg_batch)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary 
    dev_trg_mask = dev_trg_mask.view(-1)
    dev_trg_batch = dev_trg_batch.view(-1)
    dev_trg_batch = dev_trg_batch.masked_select(dev_trg_mask)
    dev_trg_mask = dev_trg_mask.unsqueeze(1).expand(len(dev_trg_mask), trg_vocab_size)
    sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
    sys_out_batch = sys_out_batch.masked_select(dev_trg_mask).view(-1, trg_vocab_size)
    loss = criterion(sys_out_batch, dev_trg_batch)
    # _, max = torch.max(sys_out_batch,dim=1)
    # print(sys_out_batch[dev_trg_batch])
    # print(max, dev_trg_batch)
    total_loss += loss
    # break
    print(total_loss, num_sents)
    print(total_loss/num_sents)
    print(torch.exp(total_loss/num_sents))


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
