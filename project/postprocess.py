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

parser = argparse.ArgumentParser(description="For postprocess.")
parser.add_argument("--file", required=True,
                    help="File to postprocess")
ret = parser.parse_known_args()
options = ret[0]

with open(options.file, 'rb') as f:
    for line in f:
        line_pp = []
        # found = False
        i = 0
        for word in line.split():
            if i == 0: 
                i += 1
                continue
            if word == '</s>':
                # found = True
                break

            # print(word)
            line_pp.append(word)
        # if not found:
            # raise ValueError()
        # print(line_pp)
        print(' '.join(line_pp).strip())
