#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import itertools
import math
from scipy.special import digamma
import pickle

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("--D", "--delta", dest="delta", default=0.01, type="float", help="Delta that defines convergence")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

f_vocab = set()

for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_vocab.add(f_i)
  if n % 500 == 0:
    sys.stderr.write(".")
sys.stderr.write("\n")

f_vocab_size = len(f_vocab)
e_count = defaultdict(lambda: f_vocab_size)
fe_count = defaultdict(lambda: 1.)
theta_fe = defaultdict(lambda: 1./f_vocab_size)
theta_e = defaultdict(lambda: 1.)
pp_diff = opts.delta + 1
alpha = 0.001
lambda_prior = 0
p0 = 0.

k = 0
perplexity = sys.maxint
# while pp_diff > opts.delta:
for _ in range(20):
  k += 1

  sys.stderr.write(str(k) + ' ')
  sys.stderr.write(str(perplexity) + '\n')

  perplexity2 = 0
  e_count = defaultdict(int)
  fe_count = defaultdict(int)

  for (iteration, (f, e)) in enumerate(bitext):
    p_fe = 1
    m = float(len(f))
    n = float(len(e))
    for i, f_i in enumerate(f):

      theta = defaultdict(int)
      Z = 0
      for j, e_j in enumerate(e):
        if j == 0:
          e_j1 = None
        else:
          e_j1 = e[j-1]

        theta[(f_i, e_j, e_j1)] = theta_fe[(f_i, e_j, e_j1)] * math.pow(math.e, - lambda_prior * abs(i/m - j/n) ) / theta_e[(e_j, e_j1)]
        Z += theta[(f_i, e_j)]
        
      p_fe *= Z

      c = theta[(f_i, None)] / Z
      fe_count[(f_i, None)] += c
      e_count[None] += c

      for e_j in set(e):
        if j == 0:
          e_j1 = None
        else:
          e_j1 = e[j-1]
        c = theta[(f_i, e_j, e_j1)] / Z
        fe_count[(f_i, e_j, e_j1)] += c
        e_count[(e_j,e_j1)] += c

    perplexity2 -= math.log(p_fe)
    if iteration % 500 == 0:
      sys.stderr.write(".")

  # theta_e = defaultdict(lambda: 0.)
  # digamma_e = defaultdict(lambda: 0.)
  # for e, val in e_count.iteritems():
  #   digamma_e[e] = math.pow(math.e, digamma(val + f_vocab_size * alpha))

  # for (f,e), val in fe_count.iteritems():
  #   a = math.pow(math.e, digamma(val + alpha)) / digamma_e[e]
  #   theta_fe[(f,e)] = a
  #   theta_e[e] += a

  theta_e = defaultdict(lambda: 0.)
  for (f,e1, e2), val in fe_count.iteritems():
    a = math.pow(math.e, digamma(val + alpha))# / math.pow(math.e, digamma(e_count[e] + f_vocab_size * alpha))
    theta_fe[(f,e1,e2)] = a
    theta_e[(e1,e2)] += a
  
  # for e, val in e_count.iteritems():
    # theta_e[e] /= math.pow(math.e, digamma(val + f_vocab_size * alpha))

  pp_diff = perplexity - perplexity2
  perplexity = perplexity2 

  if k % 10 == 0:
    f = open("e_checkpoint " + str(k/10), 'w')
    pickle.dump(e_count, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    f = open("fe_checkpoint " + str(k/10), 'w')
    pickle.dump(fe_count, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

for (f, e) in bitext:
  m = float(len(f))
  n = float(len(e))
  for (i, f_i) in enumerate(f):
    k = None
    max_prob = 0

    for (j, e_j) in enumerate(e):
      if j == 0:
        e_j1 = None
      else:
        e_j1 = e[j-1]
      theta = theta_fe[(f_i, e_j,e_j1)] * math.pow(math.e, - lambda_prior * abs(i/m - j/n) ) / theta_e[(e_j,e_j1)]
      if theta > max_prob:
        max_prob = theta
        k = j

    sys.stdout.write("%i-%i " % (i,k))
  sys.stdout.write("\n")
