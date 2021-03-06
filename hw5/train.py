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

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

  src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  batched_test_src, batched_test_src_mask, _ = utils.tensor.advanced_batchize(src_test, 1, src_vocab.stoi["<pad>"])
  batched_test_trg, batched_test_trg_mask, _ = utils.tensor.advanced_batchize(trg_test, 1, trg_vocab.stoi["<pad>"])

  batched_train_src, batched_train_src_mask, _ = utils.tensor.advanced_batchize(src_train, 1, src_vocab.stoi["<pad>"])
  batched_train_trg, batched_train_trg_mask, _ = utils.tensor.advanced_batchize(trg_train, 1, trg_vocab.stoi["<pad>"])
  batched_dev_src, batched_dev_src_mask, _ = utils.tensor.advanced_batchize(src_dev, options.batch_size, src_vocab.stoi["<pad>"])
  batched_dev_trg, batched_dev_trg_mask, _ = utils.tensor.advanced_batchize(trg_dev, options.batch_size, trg_vocab.stoi["<pad>"])

  trg_vocab_size = len(trg_vocab)
  src_vocab_size = len(src_vocab)

  # for i, batch_i in enumerate(utils.rand.srange(len(batched_train_src))):
  #     train_src_batch = Variable(batched_train_src[batch_i])  # of size (src_seq_len, batch_size)
  #     train_trg_batch = Variable(batched_train_trg[batch_i])  # of size (src_seq_len, batch_size)
  #     train_src_mask = Variable(batched_train_src_mask[batch_i])
  #     train_trg_mask = Variable(batched_train_trg_mask[batch_i])
  #     if use_cuda:
  #       train_src_batch = train_src_batch.cuda()
  #       train_trg_batch = train_trg_batch.cuda()
  #       train_src_mask = train_src_mask.cuda()
  #       train_trg_mask = train_trg_mask.cuda()


  #     sys_out_batch = nmt(train_src_batch, train_trg_batch.size()[0])
  #     # print(sys_out_batch.size())
  #     _, sys_out_batch = torch.max(sys_out_batch, dim=2)
  #     sys_out_batch = sys_out_batch.view(-1)
  #     sent = []
  #     # print(sys_out_batch)
  #     for w in sys_out_batch:
  #       # print(w)
  #       sent.append(trg_vocab.itos[w.data[0]])
  #     print(sent)

  # # Initialize encoder with weights parameters from original model
  # encoder = nn.LSTM(300, 512, bidirectional=True)

  # encoder.weight_ih_l0 = nn.Parameter(original_model['encoder.rnn.weight_ih_l0'])
  # encoder.weight_hh_l0 = nn.Parameter(original_model['encoder.rnn.weight_hh_l0'])
  # encoder.bias_ih_l0 = nn.Parameter(original_model['encoder.rnn.bias_ih_l0'])
  # encoder.bias_hh_l0 = nn.Parameter(original_model['encoder.rnn.bias_hh_l0'])

  # encoder.weight_ih_l0_reverse = nn.Parameter(original_model['encoder.rnn.weight_ih_l0_reverse'])
  # encoder.weight_hh_l0_reverse = nn.Parameter(original_model['encoder.rnn.weight_hh_l0_reverse'])
  # encoder.bias_ih_l0_reverse = nn.Parameter(original_model['encoder.rnn.bias_ih_l0_reverse'])
  # encoder.bias_hh_l0_reverse = nn.Parameter(original_model['encoder.rnn.bias_hh_l0_reverse'])

  # # Initialize decoder with weights parameters from original model
  # decoder = nn.LSTM(1324, 1024)

  # decoder.weight_ih_l0 = nn.Parameter(original_model['decoder.rnn.layers.0.weight_ih'])
  # decoder.weight_hh_l0 = nn.Parameter(original_model['decoder.rnn.layers.0.weight_hh'])
  # decoder.bias_ih_l0 = nn.Parameter(original_model['decoder.rnn.layers.0.bias_ih'])
  # decoder.bias_hh_l0 = nn.Parameter(original_model['decoder.rnn.layers.0.bias_hh'])

  # if use_cuda > 0:
  #   encoder.cuda()
  #   decoder.cuda()
  # else:
  #   encoder.cpu()
  #   decoder.cpu()

  # # Initialize embeddings
  # encoder_embedding = nn.Embedding(36616, 300)
  # decoder_embedding = nn.Embedding(23262, 300)
  # encoder_embedding.weight = nn.Parameter(original_model['encoder.embeddings.emb_luts.0.weight'])
  # decoder_embedding.weight = nn.Parameter(original_model['decoder.embeddings.emb_luts.0.weight'])

  # # Initialize Ws
  # wi = nn.Linear(1024,1024, bias=False)
  # wi.weight = nn.Parameter(original_model['decoder.attn.linear_in.weight'])

  # wo = nn.Linear(2048, 1024, bias=False)
  # wo.weight = nn.Parameter(original_model['decoder.attn.linear_out.weight'])

  # generator = nn.Linear(1024, 23262)
  # generator.weight = nn.Parameter(original_model['0.weight'])
  # generator.bias = nn.Parameter(original_model['0.bias'])

  criterion = torch.nn.NLLLoss()
  # encoder_optimizer = eval("torch.optim." + options.optimizer)(encoder.parameters(), options.learning_rate)
  # decoder_optimizer = eval("torch.optim." + options.optimizer)(decoder.parameters(), options.learning_rate)

  # soft_max = nn.Softmax()

  optimizer = eval("torch.optim." + options.optimizer)(nmt.parameters(), options.learning_rate)

  # main training loop
  last_dev_avg_loss = float("inf")
  for epoch_i in range(options.epochs):
    logging.info("At {0}-th epoch.".format(epoch_i))

    h_t_1 = Variable(torch.ones(1024))

    # srange generates a lazy sequence of shuffled range
    for i, batch_i in enumerate(utils.rand.srange(len(batched_train_src))):
      train_src_batch = Variable(batched_train_src[batch_i])  # of size (src_seq_len, batch_size)
      train_trg_batch = Variable(batched_train_trg[batch_i])  # of size (src_seq_len, batch_size)
      train_src_mask = Variable(batched_train_src_mask[batch_i])
      train_trg_mask = Variable(batched_train_trg_mask[batch_i])
      if use_cuda:
        train_src_batch = train_src_batch.cuda()
        train_trg_batch = train_trg_batch.cuda()
        train_src_mask = train_src_mask.cuda()
        train_trg_mask = train_trg_mask.cuda()

      # encoder_input = encoder_embedding(train_trg_batch)
      # sys_out_batch, (encoder_hidden_states, _) = encoder(encoder_input)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary 

      # h = Variable(torch.FloatTensor(sys_out_batch.size()[1], 1024).fill_(1./1024))
      # c = Variable(torch.FloatTensor(sys_out_batch.size()[1], 1024).fill_(0))

      # softmax = torch.nn.Softmax()
      # tanh = torch.nn.Tanh()
      # # _,w = torch.max(softmax(generator(h)), dim=1)
      # w = softmax(generator(h))

      # result = Variable(torch.FloatTensor(sys_out_batch.size()[0], sys_out_batch.size()[1], 23262))
      # for i in range(sys_out_batch.size()[0]):
      #   wht1 = wi(h).view(1, -1, 1024).expand_as(sys_out_batch)

      #   score = softmax(torch.sum(sys_out_batch * wht1, dim=2)).view(sys_out_batch.size()[0],sys_out_batch.size()[1],1)

      #   st = torch.sum(score * sys_out_batch, dim=0)
      #   ct = tanh(wo(torch.cat([st, h], dim=1)))

      #   _, w = torch.max(w, dim=1)
      #   input = torch.cat([decoder_embedding(w), ct], dim=1)
      #   input = input.view(1, input.size()[0], input.size()[1])

      #   _,(b,c) = decoder(input, (h,c))
      #   h = b[0]
      #   c = c[0]

      #   w = softmax(generator(h))
      #   result[i] = w
      # # result.append(w)
      # sys_out_batch = result
      sys_out_batch = nmt(train_src_batch, train_trg_batch.size()[0])
      # s_vector = []
      # for hs in sys_out_batch:
      #   score = hs.matmul(wi).matmul(h_t_1)
      #   score = score.unsqueeze(0)
      #   a_h_s = soft_max(score)
      #   # print a_h_s, hs.squeeze(0)
      #   s_vector.append(a_h_s.squeeze(0).dot(hs.squeeze(0)))
      # s_tilda = sum(s_vector)
      # c_t = nn.Tanh(wo.matmul(torch.cat(s_tilda, h_t_1)))

      # sys.exit()
      # train_trg_mask = train_trg_mask.view(-1)
      # train_trg_batch = train_trg_batch.view(-1)
      # train_trg_batch = train_trg_batch.masked_select(train_trg_mask)
      # train_trg_mask = train_trg_mask.unsqueeze(1).expand(len(train_trg_mask), trg_vocab_size)
      # sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
      # sys_out_batch = sys_out_batch.masked_select(train_trg_mask).view(-1, trg_vocab_size)
      print(train_trg_mask.size())
      train_trg_mask = train_trg_mask.view(-1)
      train_trg_batch = train_trg_batch.view(-1)
      train_trg_batch = train_trg_batch.masked_select(train_trg_mask)
      train_trg_mask = train_trg_mask.unsqueeze(1).expand(len(train_trg_mask), trg_vocab_size - 1)
      # print(trainin.size())
      # print(train_trg_batch[:,:-1].size())
      sys_out_batch = sys_out_batch.view(-1, trg_vocab_size - 1)
      print(trg_vocab_size)
      print(train_trg_mask.size())
      sys_out_batch = sys_out_batch.masked_select(train_trg_mask).view(-1, trg_vocab_size - 1)
      print(sys_out_batch.size())

      loss = criterion(sys_out_batch, train_trg_batch)
      logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

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

      # encoder_input = encoder_embedding(dev_trg_batch)
      # sys_out_batch = encoder(encoder_input)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary 
      sys_out_batch = nmt(dev_src_batch, dev_trg_batch.size()[0])

      dev_trg_mask = dev_trg_mask.view(-1)
      dev_trg_batch = dev_trg_batch.view(-1)
      dev_trg_batch = dev_trg_batch.masked_select(dev_trg_mask)
      dev_trg_mask = dev_trg_mask.unsqueeze(1).expand(len(dev_trg_mask), trg_vocab_size - 1)

      sys_out_batch = sys_out_batch.view(-1, trg_vocab_size - 1)
      sys_out_batch = sys_out_batch.masked_select(dev_trg_mask).view(-1, trg_vocab_size - 1)

      loss = criterion(sys_out_batch, dev_trg_batch)
      logging.debug("dev loss at batch {0}: {1}".format(batch_i, loss.data[0]))
      dev_loss += loss

    dev_avg_loss = dev_loss / len(batched_dev_in)
    logging.info("Average loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss.data[0], epoch_i))

    if (last_dev_avg_loss - dev_avg_loss).data[0] < options.estop:
      logging.info("Early stopping triggered with threshold {0} (previous dev loss: {1}, current: {2})".format(epoch_i, last_dev_avg_loss.data[0], dev_avg_loss.data[0]))
      break

    torch.save(nmt.state_dict(), open(options.model_file + ".nll_{0:.2f}.epoch_{1}".format(dev_avg_loss.data[0], epoch_i), 'wb'), pickle_module=dill)
    last_dev_avg_loss = dev_avg_loss


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
