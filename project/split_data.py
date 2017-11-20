import sys
import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW5.")
parser.add_argument("--data_file", default="data/hw5",
                    help="File prefix for training set.")
parser.add_argument("--src_lang", default="de",
                    help="Source Language. (default = words)")
parser.add_argument("--trg_lang", default="en",
                    help="Target Language. (default = phoneme)")

def main(options):

  print('Load')
  src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  print('Src Loaded')
  trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))
  print('Trg Loaded')

  src_vocab_size = len(src_vocab)

  counts = [0] * src_vocab_size
  sent = []
  num_found = 0
  for i, sentence in enumerate(src_train):
    for word in sentence:
      if counts[word] == 0:
        sent.append(i)
        num_found += 1
      counts[word] += 1
    if num_found == src_vocab_size:
      break

  sent_set = set(sent)
  sent = list(sent_set)
  english = []
  german = []

  for i, sentence in enumerate(src_train):
    if i in sent_set:
      continue
    if len(sent) == 50000 and len(english) == 50000 and len(german) == 50000:
      break
    elif len(sent) == 50000 and len(english) == 50000:
      german.append(i)
    elif len(sent) == 50000:
      english.append(i)
    else:
      sent.append(i)

  with open('src_vocab.pickle', 'wb') as f:
    dill.dump(src_vocab,f)
  with open('trg_vocab.pickle', 'wb') as f:
    dill.dump(trg_vocab,f)
  with open('src_dev.pickle', 'wb') as f:
    dill.dump(src_dev,f)
  with open('trg_dev.pickle', 'wb') as f:
    dill.dump(trg_dev,f)
  with open('src_test.pickle', 'wb') as f:
    dill.dump(src_test,f)
  with open('trg_test.pickle', 'wb') as f:
    dill.dump(trg_test,f)

  src_sents1 = []
  src_sents2 = []
  src_sents3 = []

  trg_sents1 = []
  trg_sents2 = []
  trg_sents3 = []

  for i in sent:
    src_sents1.append(src_train[i])
    trg_sents1.append(trg_train[i])

  for i in english:
    src_sents2.append(src_train[i])
    trg_sents2.append(trg_train[i])

  for i in german:
    src_sents3.append(src_train[i])
    trg_sents3.append(trg_train[i])

  with open('src_sents1.pickle', 'wb') as f:
    dill.dump(src_sents1,f)
  with open('src_sents2.pickle', 'wb') as f:
    dill.dump(src_sents2,f)
  with open('src_sents3.pickle', 'wb') as f:
    dill.dump(src_sents3,f)
  with open('trg_sents1.pickle', 'wb') as f:
    dill.dump(trg_sents1,f)
  with open('trg_sents2.pickle', 'wb') as f:
    dill.dump(trg_sents2,f)
  with open('trg_sents3.pickle', 'wb') as f:
    dill.dump(trg_sents3,f)


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
