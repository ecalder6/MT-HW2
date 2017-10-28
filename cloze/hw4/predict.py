import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
#from example_module import BiRNNLM
from model import BiGRU

import sys

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Predictor for Word Cloze.")
parser.add_argument("--data_file", required=True,
                    help="File for data set.")
parser.add_argument("--model_file", required=True,
                    help="Location to load the models.")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need

def embedding_to_word(embedding, vocab):
    return ' '.join(vocab.itos[i] for i in embedding.data)

def main(options):

    use_cuda = (len(options.gpuid) >= 1)
    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    train, dev, test, vocab = torch.load(open(options.data_file, 'rb'), pickle_module=dill)

    batched_test, batched_test_mask, _ = utils.tensor.advanced_batchize(test, 1, vocab.stoi["<pad>"])

    vocab_size = len(vocab)

    rnnlm = torch.load(options.model_file)
    if use_cuda > 0:
        rnnlm.cuda()
    else:
        rnnlm.cpu()

    rnnlm.eval()
    for test_batch, test_mask in zip(batched_test, batched_test_mask):
        test_batch = Variable(test_batch)  # of size (seq_len, batch_size)
        test_mask = Variable(test_mask)
        if use_cuda:
            test_batch = test_batch.cuda()
            test_mask = test_mask.cuda()
        sys_out_batch = rnnlm(test_batch)
        test_in_mask = test_mask.view(-1)
        test_in_mask = test_in_mask.unsqueeze(1).expand(len(test_in_mask), vocab_size)
        test_out_mask = test_mask.view(-1)
        sys_out_batch = sys_out_batch.view(-1, vocab_size)
        test_out_batch = test_batch.view(-1)
        sys_out_batch = sys_out_batch.masked_select(test_in_mask).view(-1, vocab_size)
        test_out_batch = test_out_batch.masked_select(test_out_mask)
        cur = []
        for i in range(len(test_out_batch)):
            if vocab.itos[test_out_batch.data[i]] == '<blank>':
                _, argmax = torch.max(sys_out_batch[i], 0)
                cur.append(vocab.itos[argmax.data[0]])
        print(' '.join(cur))

if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)