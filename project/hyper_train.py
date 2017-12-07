import sys
import cPickle as pickle
from pprint import pprint
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from hyperband import Hyperband

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
                    # help="If it should train with monolingual loss.")

parser.add_argument("--teacher_forcing_ratio", default=1.0, type=float,
                    help="Teacher forcing ratio.")
parser.add_argument("--mono_loss", default=0, type=int,
                    help="If it should train with monolingual loss.")
parser.add_argument("--load_file_src", default='',
                    help="Location to dump the source model.")
parser.add_argument("--load_file_trg", default='',
                    help="Location to dump the target model.")
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
parser.add_argument("--hidden_size", "-hs", default=256, type=int,
                    help="Hidden size of the LSTM. (default=256)")
parser.add_argument("--dropout", "-dr", default=0.4, type=float,
                    help="Dropout of the decoder LSTM. (default=0.4)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need
ret = parser.parse_known_args()
options = ret[0]

space = {
    'hidden_size': hp.choice('hs', (128, 256, 512)),
    'embedding_size': hp.choice('es', (100, 200, 300)),
    'dropout': hp.choice('dr', (0.0, 0.2, 0.4)),
    'teacher_forcing_ratio': hp.choice('tf', (0.0, 0.25, 0.5)),
    'mono_loss_multi': hp.choice('mlm', (0.01, 0.05, 0.1)),
    'learning_rate': hp.choice('lr', (0.001, 0.005, 0.01))
}

def handle_integers( params ):

	new_params = {}
	for k, v in params.items():
		if type( v ) == float and int( v ) == v:
			new_params[k] = int( v )
		else:
			new_params[k] = v
	
	return new_params


def get_params():
    params = sample(space)
    return handle_integers(params)

def try_params(n_iterations, params):
  n_iterations = int(n_iterations)
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
    if options.mono_loss:
      batches = batches + [(4,i) for i in range(len(batched_train_src1))]
      batches = batches + [(5,i) for i in range(len(batched_train_src1))]

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

  if os.path.isfile(options.load_file_src) and os.path.isfile(options.load_file_trg):
    src_lm = torch.load(open(options.load_file_src, 'rb'))
    trg_lm = torch.load(open(options.load_file_trg, 'rb'))
  else:
    src_lm = LM(src_vocab_size, src_vocab.stoi['<s>'], src_vocab.stoi['</s>'], params['embedding_size'], params['hidden_size'], params['dropout'], use_cuda)
    trg_lm = LM(trg_vocab_size, trg_vocab.stoi['<s>'], trg_vocab.stoi['</s>'], params['embedding_size'], params['hidden_size'], params['dropout'], use_cuda)

  if use_cuda > 0:
    src_lm.cuda()
    trg_lm.cuda()
  else:
    src_lm.cpu()
    trg_lm.cpu()

  criterion = torch.nn.NLLLoss()
  optimizer_src = eval("torch.optim." + options.optimizer)(src_lm.parameters(), params['learning_rate'])
  optimizer_trg = eval("torch.optim." + options.optimizer)(trg_lm.parameters(), params['learning_rate'])

  # main training loop
  # last_dev_avg_loss = float("inf")
  for epoch_i in range(n_iterations):
    print(epoch_i)
    logging.info("At {0}-th epoch.".format(epoch_i))

    shuffle(batches)
    src_lm.train()
    trg_lm.train()
    for i, (index, batch_i) in enumerate(batches):

      train_src_batch = None
      train_src_mask = None
      train_trg_batch = None
      train_trg_mask = None

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
        train_trg_batch = Variable(batched_train_trg2[batch_i])
        train_trg_mask = Variable(batched_train_trg_mask2[batch_i])
        if use_cuda:
          train_trg_batch = train_trg_batch.cuda()
          train_trg_mask = train_trg_mask.cuda()
      elif index == 3:
        train_src_batch = Variable(batched_train_src3[batch_i])
        train_src_mask = Variable(batched_train_src_mask3[batch_i])
        if use_cuda:
          train_src_batch = train_src_batch.cuda()
          train_src_mask = train_src_mask.cuda()
      elif index == 4:
        train_src_batch = Variable(batched_train_src1[batch_i])
        train_src_mask = Variable(batched_train_src_mask1[batch_i])
        if use_cuda:
          train_src_batch = train_src_batch.cuda()
          train_src_mask = train_src_mask.cuda()
      elif index == 5:
        train_trg_batch = Variable(batched_train_trg1[batch_i])
        train_trg_mask = Variable(batched_train_trg_mask1[batch_i])
        if use_cuda:
          train_trg_batch = train_trg_batch.cuda()
          train_trg_mask = train_trg_mask.cuda()
      else:
        raise ValueError()

      total_loss = 0
      if index == 1:
        optimizer_trg.zero_grad()
        optimizer_src.zero_grad()
        h_src, c_src = src_lm(sent=train_src_batch)
        use_teacher_forcing = True if random.random() < params['teacher_forcing_ratio'] else False
        sys_out_batch = trg_lm(h=h_src, c=c_src, encode=False, tgt_sent=train_trg_batch, teacher_forcing=use_teacher_forcing)

        train_trg_mask_tmp = train_trg_mask.view(-1)
        train_trg_batch_tmp = train_trg_batch.view(-1)
        train_trg_batch_tmp = train_trg_batch_tmp.masked_select(train_trg_mask_tmp)
        train_trg_mask_tmp = train_trg_mask_tmp.unsqueeze(1).expand(len(train_trg_mask_tmp), trg_vocab_size)
        sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
        sys_out_batch = sys_out_batch.masked_select(train_trg_mask_tmp).view(-1, trg_vocab_size)
        loss = criterion(sys_out_batch, train_trg_batch_tmp)
        loss.backward()
        optimizer_src.step()
        optimizer_trg.step()
        if i % 100 == 0:
          logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))

      elif options.mono_loss and train_src_batch is not None:
        optimizer_trg.zero_grad()
        optimizer_src.zero_grad()
        h_src, c_src = src_lm(sent=train_src_batch)
        use_teacher_forcing = True if random.random() < params['teacher_forcing_ratio'] else False
        sys_out_batch = src_lm(h=h_src, c=c_src, encode=False, tgt_sent=train_src_batch, teacher_forcing=use_teacher_forcing)

        train_src_mask_tmp = train_src_mask.view(-1)
        train_src_batch_tmp = train_src_batch.view(-1)
        train_src_batch_tmp = train_src_batch_tmp.masked_select(train_src_mask_tmp)
        train_src_mask_tmp = train_src_mask_tmp.unsqueeze(1).expand(len(train_src_mask_tmp), src_vocab_size)
        sys_out_batch = sys_out_batch.view(-1, src_vocab_size)
        sys_out_batch = sys_out_batch.masked_select(train_src_mask_tmp).view(-1, src_vocab_size)
        loss = criterion(sys_out_batch, train_src_batch_tmp)
        loss *= params['mono_loss_multi'] * (1.0 / 10 * epoch_i)
        loss.backward()
        optimizer_src.step()
        optimizer_trg.step()
        if i % 100 == 0:
          logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))
      
      elif train_trg_batch is not None and options.mono_loss:
        optimizer_trg.zero_grad()
        optimizer_src.zero_grad()

        h_trg, c_trg = trg_lm(sent=train_trg_batch)
        use_teacher_forcing = True if random.random() < params['teacher_forcing_ratio'] else False
        sys_out_batch = trg_lm(h=h_trg, c=c_trg, encode=False, tgt_sent=train_trg_batch, teacher_forcing=use_teacher_forcing)

        train_trg_mask_tmp = train_trg_mask.view(-1)
        train_trg_batch_tmp = train_trg_batch.view(-1)
        train_trg_batch_tmp = train_trg_batch_tmp.masked_select(train_trg_mask_tmp)
        train_trg_mask_tmp = train_trg_mask_tmp.unsqueeze(1).expand(len(train_trg_mask_tmp), trg_vocab_size)
        sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
        sys_out_batch = sys_out_batch.masked_select(train_trg_mask_tmp).view(-1, trg_vocab_size)
        loss = criterion(sys_out_batch, train_trg_batch_tmp)
        loss *= params['mono_loss_multi'] * (1.0 / 10 * epoch_i)
        loss.backward()
        optimizer_src.step()
        optimizer_trg.step()
        if i % 100 == 0:
          logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))

    # validation -- this is a crude esitmation because there might be some paddings at the end
    dev_loss = 0.0
    src_lm.eval()
    trg_lm.eval()
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

    # torch.save(src_lm, open(options.model_file_src + ".nll_{0:.2f}.epoch_{1}".format(dev_avg_loss.data[0], epoch_i), 'wb'), pickle_module=dill)
    # torch.save(trg_lm, open(options.model_file_trg + ".nll_{0:.2f}.epoch_{1}".format(dev_avg_loss.data[0], epoch_i), 'wb'), pickle_module=dill)
    # last_dev_avg_loss = dev_avg_loss

    return {'loss': dev_avg_loss.data[0]}

try:
	output_file = sys.argv[1]
	if not output_file.endswith( '.pkl' ):
		output_file += '.pkl'	
except IndexError:
	output_file = 'results.pkl'
	
print "Will save results to", output_file

#

hb = Hyperband( get_params, try_params )
results = hb.run( skip_last = 1 )

print "{} total, best:\n".format( len( results ))

for r in sorted( results, key = lambda x: x['loss'] )[:5]:
	print "loss: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format( 
		r['loss'], r['seconds'], r['iterations'], r['counter'] )
	pprint( r['params'] )
	print

print "saving..."

with open( output_file, 'wb' ) as f:
    pickle.dump( results, f )