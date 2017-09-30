#!/usr/bin/env python
from __future__ import print_function
import optparse
import sys
from collections import defaultdict
import itertools
import math
import pickle
from random import shuffle


import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-a", "--alignment", dest="alignment", default="new_submit.txt", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("--D", "--delta", dest="delta", default=1.0, type="float", help="Delta that defines convergence")

(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

f_char_set = set()
e_char_set = set()

for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    for c in f_i:
      f_char_set.add(c)
  for e_i in set(e):
    for c in e_i:
      e_char_set.add(c)
  if n % 500 == 0:
    sys.stderr.write(".")
sys.stderr.write("\n")

f_char_set = list(f_char_set)
e_char_set = list(e_char_set)

align_pairs = []
for sentence in open(opts.alignment):
  align_pair = []
  for word in sentence.strip().split():
    [a,b] = word.split('-')
    align_pair.append((a,b))
  align_pairs.append(align_pair)

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    self.e_char_dict = dict()
    self.f_char_dict = dict()
    for n, e_word in enumerate(e_char_set):
      self.e_char_dict[e_word] = n

    for n, f_word in enumerate(f_char_set):
      self.f_char_dict[f_word] = n

    self.encoding_size = 32
    self.embed_e = nn.Embedding(len(e_char_set),len(e_char_set))
    self.embed_f = nn.Embedding(len(f_char_set),len(f_char_set))
    self.e_word_lstm = nn.LSTM(len(e_char_set), self.encoding_size, 1)
    self.f_word_lstm = nn.LSTM(len(f_char_set), self.encoding_size, 1)
    self.e_lstm = nn.LSTM(self.encoding_size, self.encoding_size, 1, bidirectional=True)
    self.f_lstm = nn.LSTM(self.encoding_size, self.encoding_size, 1, bidirectional=True)
    self.tanh = nn.Sigmoid()
    self.conv = nn.Conv2d(2,1,3,padding=1)
    #  _, (h,_) = rnn(input)

  def forward(self, f, e, diag):
    e_seq = Variable(torch.FloatTensor(len(e), self.encoding_size))
    for n, word in enumerate(e):
      e_word = []
      for c in word:
        e_word.append(self.e_char_dict[c])
      embedding = self.embed_e(Variable(torch.LongTensor([e_word])))
      embedding = embedding.view(len(word), 1, len(e_char_set))
      _,(h,_) = self.e_word_lstm(embedding)
      e_seq[n] = h.view(self.encoding_size)

    f_seq = Variable(torch.FloatTensor(len(f), self.encoding_size))
    for n, word in enumerate(f):
      f_word = []
      for c in word:
        f_word.append(self.f_char_dict[c])
      embedding = self.embed_f(Variable(torch.LongTensor([f_word])))
      embedding = embedding.view(len(word), 1, len(f_char_set))
      _,(h,_) = self.f_word_lstm(embedding)
      f_seq[n] = h.view(self.encoding_size)

    e_output, _ = self.e_lstm(e_seq.view(len(e),1,self.encoding_size))
    e_seq = self.tanh(e_output)
    e_seq = e_seq.view(len(e), 2*self.encoding_size)

    f_output, _ = self.f_lstm(f_seq.view(len(f),1,self.encoding_size))
    f_seq = self.tanh(f_output)
    f_seq = f_seq.view(len(f), 2*self.encoding_size)

    score_ij = Variable(torch.FloatTensor(len(f),len(e)))
    for i in range(len(f)):
      score_j = []
      for j in range(len(e)):
        score_ij[i,j] = f_seq[i].dot(e_seq[j])

    return score_ij
    # score_ij = score_ij.view(1, len(f), len(e))
    # diag = diag.view(1, len(f), len(e))
    
    # score_ij = torch.cat([score_ij, diag], 0).view(1, 2, len(f), len(e))
    # return self.conv(score_ij).view(len(f),len(e))

  def evaluate(self):
    scores_count = defaultdict(int)
    scores = defaultdict(int)
    scores_var = defaultdict(int)

    for f,e in bitext:
      score_ij = self.forward(f,e, make_diag(len(f), len(e)))
      for i, word_f in enumerate(f):
        for j, word_e in enumerate(e):
          scores[word_e] += score_ij[i,j]
          scores_var[word_e] += score_ij[i,j] ** 2
          scores_count[word_e] += 1

    for f,e in bitext:
      score_ij = self.forward(f,e, make_diag(len(f), len(e)))
      for i, word_f in enumerate(f):
        for j, word_e in enumerate(e):
          if score_ij[i,j].data[0] > ( (scores[word_e] / scores_count[word_e]) + ((scores_var[word_e] - scores[word_e] ** 2)/scores_count[word_e]).sqrt()).data[0]:
            # print(score_ij[i,j], ( (scores[word_e] / scores_count[word_e]) + ((scores_var[word_e] - scores[word_e] ** 2)/scores_count[word_e]).sqrt()))
            sys.stdout.write("%i-%i " % (i,j))
      sys.stdout.write("\n")


def loss_fn_1(score, align_pair):
  # loss = torch.sum(score)
  count = 0
  loss = 0
  for i, row in enumerate(score):
    for j, val in enumerate(row):
      count += 1
      if (str(i),str(j)) in align_pair:
        # print(i,j, val)
        # loss += (1. + (-1 * val).exp()).log()
        loss += 1./ score[i,j]
      else:
        loss += score[i,j]
        # loss += 10*(1. + (val).exp()).log()
  # loss = 0
  # for x,y in align_pair:
    # print(x,y)
    # loss += 100*(1. + (-1 * score[int(x),int(y)]).exp()).log()
  print(loss)
  return loss

def loss_fn_2(score):
  y = 0
  i = 0
  for row in score:
    z = 0
    for val in row:
      i += 1
      z += val.exp()
    y += (1. + 1./ z).log()

  for j in range(len(score[0])):
      z = 0.
      for i in range(len(score)):
          z += score[i,j].exp()
      y += (1. + 1./z).log()

  return i * y

def loss_fn_3(score):
  y = 0
  i = 0
  for row in score:
    z = 0
    for val in row:
      i += 1
      z += val.exp()
    y += (1. + z).log()

  for j in range(len(score[0])):
      z = 0.
      for i in range(len(score)):
          z += score[i,j].exp()
      y += (1. + z).log()

  return i*y

def make_diag(m, n):
  m_f = float(m)
  n_f = float(n)
  prior_ij = []
  for i in range(m):
    prior_j = []
    for j in range(n):
      prior_j.append(abs(i/m_f - j/n_f))
    prior_ij.append(prior_j)
  return Variable(torch.FloatTensor(prior_ij))

model = Model()
shuffle_index = range(len(bitext))
# shuffle(shuffle_index)

model.zero_grad()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

iterations = 0
for _ in range(50):
  
  for n, i in enumerate(shuffle_index):
    # model.eval()
    # model.evaluate()
    model.train()
    # model.evaluate()
    model.zero_grad()
    print(i)
    e = bitext[i][1]
    f = bitext[i][0]
    diag = make_diag(len(f), len(e))
    score = model(f,e, diag)
    # print(score)
    y1 = loss_fn_1(score, align_pairs[i])
    # y1 = loss_fn_2(score)
    # y1.backward(retain_graph=True)
    # y2 = y1 #+ loss_fn_3(score)
    # y2 = y1
    y1.backward()

    # f2 = bitext[shuffle_index[(n+1)%len(shuffle_index)]][0]
    # diag = make_diag(len(f2), len(e))
    # score = model(f2,e, diag)
    # print(score)
    # y2 = loss_fn_3(score)
    # y2.backward()
    optimizer.step()
 
    # print(n)
  model.eval()
  model.evaluate()

