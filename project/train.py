import utils.tensor
import utils.rand

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

parser.add_argument("--model_file", required=True,
                    help="Location to dump the models.")
parser.add_argument("--batch_size", default=1, type=int,
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
parser.add_argument("--embedding_size", "-es", default=300, type=int,
                    help="Embedding size of the LSTM. (default=300)")
parser.add_argument("--hidden_size", "-hs", default=512, type=int,
                    help="Hidden size of the LSTM. (default=512)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need

def main(options):

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

  src_vocab = dill.load(open('src_vocab.pickle', 'rb'))
  trg_vocab = dill.load(open('trg_vocab.pickle', 'rb'))

  src_dev = dill.load(open('src_dev.pickle', 'rb'))
  trg_dev = dill.load(open('trg_dev.pickle', 'rb'))
  batched_dev_src, batched_dev_src_mask, sort_index = utils.tensor.advanced_batchize(src_dev, options.batch_size, src_vocab.stoi["<blank>"])
  batched_dev_trg, batched_dev_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_dev, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)
  
  batches = []

  if options.contain_bilingual:
    print('Load')
    src_train = dill.load(open('src_sents1.pickle', 'rb'))
    print('Load src sents 1')
    trg_train = dill.load(open('trg_sents1.pickle', 'rb'))
    print('Load trg sents 1')
    batched_train_src1, batched_train_src_mask1, sort_index = utils.tensor.advanced_batchize(src_train, options.batch_size, src_vocab.stoi["<blank>"])
    batched_train_trg1, batched_train_trg_mask1 = utils.tensor.advanced_batchize_no_sort(trg_train, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)
    batches = batches + [(1,i) for i in range(len(batched_train_src1))]

  if options.contain_trg:
    print('Load')
    # src_train = dill.load(open('src_sents2.pickle', 'rb'))
    # print('Load src sents 2')
    trg_train = dill.load(open('trg_sents2.pickle', 'rb'))
    print('Load trg sents 2')
    # batched_train_src2, batched_train_src_mask2, sort_index = utils.tensor.advanced_batchize(src_train, options.batch_size, src_vocab.stoi["<blank>"])
    batched_train_trg2, batched_train_trg_mask2 = utils.tensor.advanced_batchize_no_sort(trg_train, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)
    batches = batches + [(2,i) for i in range(len(batched_train_trg2))]

  if options.contain_src:
    print('Load')
    src_train = dill.load(open('src_sents3.pickle', 'rb'))
    print('Load src sents 3')
    # trg_train = dill.load(open('trg_sents3.pickle', 'rb'))
    # print('Load trg sents 3')
    batched_train_src3, batched_train_src_mask3, sort_index = utils.tensor.advanced_batchize(src_train, options.batch_size, src_vocab.stoi["<blank>"])
    # batched_train_trg3, batched_train_trg_mask3 = utils.tensor.advanced_batchize_no_sort(trg_train, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)
    batches = batches + [(3,i) for i in range(len(batched_train_src3))]

  src_vocab_size = len(src_vocab)
  trg_vocab_size = len(trg_vocab)

  src_lm = LM(src_vocab_size, src_vocab.stoi['<s>'], src_vocab.stoi['</s>'], options.embedding_size, options.hidden_size, use_cuda)
  trg_lm = LM(trg_vocab_size, trg_vocab.stoi['<s>'], trg_vocab.stoi['</s>'], options.embedding_size, options.hidden_size, use_cuda)
  # src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  # trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  # batched_train_src, batched_train_src_mask, sort_index = utils.tensor.advanced_batchize(src_train, options.batch_size, src_vocab.stoi["<blank>"])
  # batched_train_trg, batched_train_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_train, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)
  # batched_dev_src, batched_dev_src_mask, sort_index = utils.tensor.advanced_batchize(src_dev, options.batch_size, src_vocab.stoi["<blank>"])
  # batched_dev_trg, batched_dev_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_dev, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)

  # trg_vocab_size = len(trg_vocab)

  # original_model = torch.load(open(options.original_model_file, 'rb'))
  # nmt = NMT(original_model, use_cuda) # TODO: add more arguments as necessary 
  if use_cuda > 0:
    src_lm.cuda()
    trg_lm.cuda()
  else:
    src_lm.cpu()
    trg_lm.cpu()

  criterion = torch.nn.NLLLoss()
  optimizer_src = eval("torch.optim." + options.optimizer)(src_lm.parameters(), options.learning_rate)
  optimizer_trg = eval("torch.optim." + options.optimizer)(trg_lm.parameters(), options.learning_rate)

  # main training loop
  # last_dev_avg_loss = float("inf")
  for epoch_i in range(options.epochs):
    print(epoch_i)
    logging.info("At {0}-th epoch.".format(epoch_i))
    # srange generates a lazy sequence of shuffled range

    shuffle(batches)
    for i, (index, batch_i) in enumerate(batches):
      optimizer_trg.zero_grad()
      optimizer_src.zero_grad()

      if index == 1:
        train_src_batch = Variable(batched_train_src1[batch_i])
        train_src_mask = Variable(batched_train_src_mask1[batch_i])
        train_trg_batch = Variable(batched_train_trg1[batch_i])
        train_trg_mask = Variable(batched_train_trg_mask1[batch_i])
        if use_cuda:
          train_src_batch = train_src_batch.cuda()
          train_trg_batch = train_trg_batch.cuda()
          train_src_mask = train_src_mask.cuda()
          train_trg_mask = train_trg_mask.cuda()
      elif index == 2:
        train_src_batch = None
        train_src_mask = None
        train_trg_batch = Variable(batched_train_trg2[batch_i])
        train_trg_mask = Variable(batched_train_trg_mask2[batch_i])
        if use_cuda:
          train_trg_batch = train_trg_batch.cuda()
          train_trg_mask = train_trg_mask.cuda()
      elif index == 3:
        train_src_batch = Variable(batched_train_src3[batch_i])
        train_src_mask = Variable(batched_train_src_mask3[batch_i])
        train_trg_batch = None
        train_trg_mask = None
        if use_cuda:
          train_src_batch = train_src_batch.cuda()
          train_src_mask = train_src_mask.cuda()

      if train_src_batch is not None:
        h_src, c_src = src_lm(sent=train_src_batch)
      elif train_trg_batch is not None and options.mono_loss:
        h_trg, c_trg = trg_lm(sent=train_src_batch)
      else:
        continue

      if index == 1:

        use_teacher_forcing = True if random.random() < options.teacher_forcing_ratio else False
        sys_out_batch = trg_lm(h=h_src, c=c_src, encode=False, tgt_sent=train_trg_batch, teacher_forcing=use_teacher_forcing)

        train_trg_mask = train_trg_mask.view(-1)
        train_trg_batch = train_trg_batch.view(-1)
        train_trg_batch = train_trg_batch.masked_select(train_trg_mask)
        train_trg_mask = train_trg_mask.unsqueeze(1).expand(len(train_trg_mask), trg_vocab_size)
        sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
        sys_out_batch = sys_out_batch.masked_select(train_trg_mask).view(-1, trg_vocab_size)
        loss = criterion(sys_out_batch, train_trg_batch)
        logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))
        loss.backward()

      if options.mono_loss:
        if train_src_batch is not None:
          use_teacher_forcing = True if random.random() < options.teacher_forcing_ratio else False
          sys_out_batch = trg_lm(h=h_src, c=c_src, encode=False, tgt_sent=train_src_batch, teacher_forcing=use_teacher_forcing)

          train_src_mask = train_src_mask.view(-1)
          train_src_batch = train_src_batch.view(-1)
          train_src_batch = train_src_batch.masked_select(train_src_mask)
          train_src_mask = train_src_mask.unsqueeze(1).expand(len(train_src_mask), src_vocab_size)
          sys_out_batch = sys_out_batch.view(-1, src_vocab_size)
          sys_out_batch = sys_out_batch.masked_select(train_src_mask).view(-1, src_vocab_size)
          loss = criterion(sys_out_batch, train_src_batch)
          logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))
          loss.backward()

        if train_trg_batch is not None:
          use_teacher_forcing = True if random.random() < options.teacher_forcing_ratio else False
          sys_out_batch = trg_lm(h=h_trg, c=c_trg, encode=False, tgt_sent=train_trg_batch, teacher_forcing=use_teacher_forcing)

          train_trg_mask = train_trg_mask.view(-1)
          train_trg_batch = train_trg_batch.view(-1)
          train_trg_batch = train_trg_batch.masked_select(train_trg_mask)
          train_trg_mask = train_trg_mask.unsqueeze(1).expand(len(train_trg_mask), trg_vocab_size)
          sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
          sys_out_batch = sys_out_batch.masked_select(train_trg_mask).view(-1, trg_vocab_size)
          loss = criterion(sys_out_batch, train_trg_batch)
          logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))
          loss.backward()

      optimizer_src.step()
      optimizer_trg.step()

    # validation -- this is a crude esitmation because there might be some paddings at the end
    dev_loss = 0.0
    for batch_i in range(len(batched_dev_src)):
      dev_src_batch = Variable(batched_dev_src[batch_i], volatile=True)
      dev_trg_batch = Variable(batched_dev_trg[batch_i], volatile=True)
      dev_src_mask = Variable(batched_dev_src_mask[batch_i], volatile=True)
      dev_trg_mask = Variable(batched_dev_trg_mask[batch_i], volatile=True)
      if use_cuda:
        dev_src_batch = dev_src_batch.cuda()
        dev_trg_batch = dev_trg_batch.cuda()
        dev_src_mask = dev_src_mask.cuda()
        dev_trg_mask = dev_trg_mask.cuda()

      h_src, c_src = src_lm(sent=dev_src_batch)
      sys_out_batch = trg_lm(h=h_src, c=c_src, encode=False, tgt_sent=dev_trg_batch)

      dev_trg_mask = dev_trg_mask.view(-1)
      dev_trg_batch = dev_trg_batch.view(-1)
      dev_trg_batch = dev_trg_batch.masked_select(dev_trg_mask)
      dev_trg_mask = dev_trg_mask.unsqueeze(1).expand(len(dev_trg_mask), trg_vocab_size)
      sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
      sys_out_batch = sys_out_batch.masked_select(dev_trg_mask).view(-1, trg_vocab_size)

      loss = criterion(sys_out_batch, dev_trg_batch)
      logging.debug("dev loss at batch {0}: {1}".format(batch_i, loss.data[0]))
      dev_loss += loss

    dev_avg_loss = dev_loss / len(batched_dev_src)
    logging.info("Average loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss.data[0], epoch_i))

    # if (last_dev_avg_loss - dev_avg_loss).data[0] < options.estop:
      # logging.info("Early stopping triggered with threshold {0} (previous dev loss: {1}, current: {2})".format(epoch_i, last_dev_avg_loss.data[0], dev_avg_loss.data[0]))
      # break

    torch.save(nmt, open(options.model_file + ".nll_{0:.2f}.epoch_{1}".format(dev_avg_loss.data[0], epoch_i), 'wb'), pickle_module=dill)
    # last_dev_avg_loss = dev_avg_loss


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
