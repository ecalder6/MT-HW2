#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import itertools
import math
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
#read from file to initialize our e_count and fe_count
f = open("e_checkpoint 1")
e_count = pickle.load(f)
f.close()
f = open("fe_checkpoint 1")
fe_count = pickle.load(f)
f.close()


perplexity = sys.maxint
pp_diff = opts.delta + 1

i = 0
while pp_diff > opts.delta:
  i += 1

  sys.stderr.write(str(i) + ' ')
  sys.stderr.write(str(perplexity) + '\n')

  perplexity2 = 0
  e_count2 = defaultdict(int)
  fe_count2 = defaultdict(int)
  for (n, (f, e)) in enumerate(bitext):
    p_fe = 1
    for f_i in set(f):
      Z = 0.
      theta = defaultdict(int)
      for e_j in set(e):
        theta[(f_i, e_j)] = fe_count[(f_i, e_j)] / e_count[e_j]
        Z += theta[(f_i, e_j)]
      p_fe *= Z

      for e_j in set(e):
        c = theta[(f_i, e_j)] / Z
        fe_count2[(f_i, e_j)] += c
        e_count2[e_j] += c
    perplexity2 += -math.log(p_fe)
    if n % 500 == 0:
      sys.stderr.write(".")
  e_count = e_count2
  fe_count = fe_count2
  pp_diff = perplexity - perplexity2
  print perplexity, perplexity2
  print pp_diff
  perplexity = perplexity2 

for (f, e) in bitext:
  for (i, f_i) in enumerate(f):
    k = 0
    max_prob = 0
    for (j, e_j) in enumerate(e):
      theta_fe = fe_count[(f_i, e_j)] / e_count[e_j]
      if theta_fe > max_prob:
        max_prob = theta_fe
        k = j
    sys.stdout.write("%i-%i " % (i,k))
  sys.stdout.write("\n")
