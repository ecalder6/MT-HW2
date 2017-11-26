import utils.tensor
import utils.rand

import os.path
import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from model import LM

from random import shuffle
import random

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW5.")
# parser.add_argument("--parallel_file", default="",
                    # help="File prefix for training set.")
# parser.add_argument("--src_lang", default="de",
                    # help="Source Language. (default = de)")
# parser.add_argument("--trg_lang", default="en",
                    # help="Target Language. (default = en)")
parser.add_argument("--contain_bilingual", default=1, type=int,
                    help="If it should train on bilingual data.")
parser.add_argument("--contain_trg", default=0, type=int,
                    help="If it should train on target (english) monolingual data.")
parser.add_argument("--contain_src", default=0, type=int,
                    help="If it should train on source (german) monolingual data.")
parser.add_argument("--mono_loss", default=0, type=int,
                    help="If it should train with monolingual loss.")

parser.add_argument("--teacher_forcing_ratio", default=1.0, type=float,
                    help="Teacher forcing ratio.")

parser.add_argument("--model_file_src", required=True,
                    help="Location to dump the source model.")
parser.add_argument("--model_file_trg", required=True,
                    help="Location to dump the target model.")

# feel free to add more arguments as you need

def main(options):

  # use_cuda = (len(options.gpuid) >= 1)
  # if options.gpuid:
    # cuda.set_device(options.gpuid[0])

  # src_lm = torch.load(options.model_file_src, pickle_module=dill)
  # trg_lm = torch.load(options.model_file_trg, pickle_module=dill)

  src_lm = torch.load(open(options.model_file_src, 'rb'))
  trg_lm = torch.load(open(options.model_file_trg, 'rb'))

  src_vocab = dill.load(open('src_vocab.pickle', 'rb'))
  trg_vocab = dill.load(open('trg_vocab.pickle', 'rb'))

  src_test = dill.load(open('src_test.pickle', 'rb'))
  trg_test = dill.load(open('trg_test.pickle', 'rb'))

  # src_lm = LM(len(src_vocab), src_vocab.stoi['<s>'], src_vocab.stoi['</s>'], 300, 512)
  # trg_lm = LM(len(trg_vocab), trg_vocab.stoi['<s>'], trg_vocab.stoi['</s>'], 300, 512)

  for i, sent in enumerate(src_test):
    # print(sent.size())
    sent = Variable(sent, volatile=True).view(-1, 1).cuda()
    trg_sent = Variable(trg_test[i], volatile=True).view(-1,1).cuda()
    h,c = src_lm(sent=sent)
    # print(h,c)
    results = trg_lm(h=h, c=c, encode=False, tgt_sent=trg_sent, teacher_forcing=True)
    # print(results)
    # print(results.size())
    _,w = torch.max(results.view(trg_sent.size()[0], -1), dim=1)
    sentence = []
    for word in w:
      sentence.append(trg_vocab.itos[word.data[0]])
    print(' '.join(sentence).encode('utf-8').strip())
    # print(h.size())
    # print(c.size())


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
