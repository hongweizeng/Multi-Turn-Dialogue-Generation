#!/usr/bin/env python

from __future__ import division

import argparse
import os
import sys
import random

import torch
from torch import cuda

import mtdg.opts as opts
import mtdg.utils
from mtdg.utils.logging import logger
from mtdg.generator import build_generator, Generator




def main(opt):
    generator = build_generator(opt)
    generator.generate(data_path=opt.data, batch_size=opt.batch_size)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train.py')
    opts.generate_opts(parser)
    opt = parser.parse_args()

    main(opt)