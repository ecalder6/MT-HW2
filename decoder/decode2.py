#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
import math

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=5, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

def get_neighbors(h, tm, obj):

  #english, french, logprob, start, end
  hypotheses = []
  
  for i in xrange(0, len(h)):
    
    #swap
    if i < len(h) -1:
      new_hyp = list(h)
      new_hyp[i], new_hyp[i+1] = new_hyp[i+1], new_hyp[i]
      hypotheses.append(new_hyp)

    #replacement
    for translation in tm[h[i][1]]:
      if translation[0] != h[i][0]:
        new_hyp = list(h)
        new_hyp[i] = obj(translation[0], h[i][1], translation[1], h[i][3], h[i][4])
        hypotheses.append(new_hyp)
        
        #bi-replace
        for j in xrange(i + 1, len(h)):
          for translation2 in tm[h[j][1]]:
            if translation2[0] != h[j][0]:
              second_hyp = list(new_hyp)
              second_hyp[j] = obj(translation2[0], h[j][1], translation2[1],
                  h[j][3], h[i][4])
              hypotheses.append(second_hyp)

    
    #merge neighbors
    if i < len(h)- 1 and h[i][1] + h[i+1][1] in tm:
      for translation in tm[h[i][1] + h[i+1][1]]:
        new_hyp = list(h)
        del new_hyp[i+1]
        new_hyp[i] = obj(translation[0], h[i][1] + h[i+1][1], translation[1], h[i][3], h[i+1][4])

    #split
    if len(h[i][1]) > 1:
      first_half = h[i][1][0:len(h[i][1])/2]
      second_half = h[i][1][len(h[i][1])/2:]
      if first_half in tm and second_half in tm:
        for translation1 in tm[first_half]:
          obj1 = obj(translation1[0], first_half, translation1[1], h[i][3], h[i][3]
                  + len(first_half) - 1)
          for translation2 in tm[second_half]:
            obj2 = obj(translation2[0], second_half, translation2[1], obj1[4] + 1,
                    obj1[4] + len(second_half))
            new_hyp = h[0:i] + [obj1] + [obj2] + h[i+1:]
            hypotheses.append(new_hyp)
  return hypotheses

def score(l, tm, lambda_lm = 1, lambda_tm = 1, lambda_w = 1, lambda_d = 1, alpha = .8):
  state = lm.begin()
  e_logprob = 0.
  fe_logprob = 0.
  reordering_logprob = 0.
  for i, obj in enumerate(l):
    fe_logprob += obj.logprob

    for word in obj.english.split():
      (state, word_logprob) = lm.score(state, word)
      e_logprob += word_logprob

    if i == 0:
      reordering_logprob += abs(obj.start - (-1) - 1) * math.log(0.9)
    else:
      reordering_logprob += abs(obj.start - (l[i-1].end) - 1) * math.log(0.9)

  e_logprob += lm.end(state)
  return e_logprob + fe_logprob + reordering_logprob

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
i = 0


# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, french")
  initial_hypothesis = hypothesis(0.0, (lm.begin(),-1), None, None, None)
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
      for j in xrange(i+1,len(f)+1):
        if f[i:j] in tm:

          for phrase in tm[f[i:j]]:

            logprob = h.logprob + phrase.logprob
            lm_state = h.lm_state
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score2(lm_state, word)
              logprob += word_logprob

            # add 'reordering'
            logprob += abs(i - lm_state[1] - 1) * math.log(0.9)
            lm_state = (lm_state[0], j - 1)

            logprob += lm.end2(lm_state) if j == len(f) else 0.0
            new_hypothesis = hypothesis(logprob, lm_state, h, phrase, f[i:j])
            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
              stacks[j][lm_state] = new_hypothesis 

        for k in xrange(i + 1, j):
          if f[i:k] in tm and f[k:j] in tm:
            for phrase1 in tm[f[i:k]]:
              for phrase2 in  tm[f[k:j]]:

                logprob1 = h.logprob + phrase2.logprob
                lm_state1 = h.lm_state
                for word in phrase2.english.split():
                  (lm_state1, word_logprob) = lm.score2(lm_state1, word)
                  logprob1 += word_logprob
                logprob1 += abs(k - lm_state[1] - 1) * math.log(0.9)
                lm_state1 = (lm_state1[0], j - 1)

                logprob2 = logprob1 + phrase1.logprob
                lm_state2 = lm_state1
                for word in phrase1.english.split():
                  (lm_state2, word_logprob) = lm.score2(lm_state2, word)
                  logprob2 += word_logprob


                # add reordering term
                # logprob += abs(k - lm_state[1] - 1) * math.log(0.9)
                logprob2 += abs(i - (j-1) - 1) * math.log(0.9)
                lm_state2 = (lm_state2[0], k - 1)

                logprob2 += lm.end2(lm_state2) if j == len(f) else 0.0
                new_hypothesis = hypothesis(logprob2, lm_state2, hypothesis(logprob1, lm_state1, h, phrase2, f[k:j]), phrase1, f[i:k])
                if lm_state2 not in stacks[j] or stacks[j][lm_state2].logprob < logprob2: # second case is recombination
                  stacks[j][lm_state2] = new_hypothesis

  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  h = winner
  list_version = []
  obj = namedtuple("obj", "english, french, logprob, start, end")
  while h is not None:
    if h.phrase is None:
      break
    list_version.append(obj(h.phrase.english, h.french, h.phrase.logprob, h.lm_state[1] + 1 - len(h.french), h.lm_state[1]))
    h = h.predecessor
  list_version = list_version[::-1]

  # Greedy
  current = list_version
  while True:
    s_current = score(current, tm)
    s = s_current
    best = None
    for h in get_neighbors(current, tm, obj):
      c = score(h, tm)
      if c > s:
        s = c
        best = h
    if s == s_current:
      break
    current = best
  winner = current
  def extract_english(h): 
    return ' '.join([t.english for t in h])

  # def extract_english(h): 
  #   return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
