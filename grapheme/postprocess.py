from __future__ import print_function
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


with open('result.txt', 'rb') as f:
    for line in f:
        line_pp = []
        found = False
        for word in line.split():
            if word == '</s>':
                found = True
                break
            # print(word)
            line_pp.append(word)
        # if not found:
            # raise ValueError()
        print(' '.join(line_pp).encode('utf-8').strip())
