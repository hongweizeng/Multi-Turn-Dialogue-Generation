#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import os
import glob
import sys

from collections import Counter, defaultdict
from itertools import chain, count
from pathlib import Path

import torch
import torchtext

import mtdg
import mtdg.data
import mtdg.inputters.text_dataset
import mtdg.opts as opts

project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('data/')
ubuntu_dir = datasets_dir.joinpath('ubuntu/')

ubuntu_meta_dir = ubuntu_dir.joinpath('meta/')
dialogs_dir = ubuntu_dir.joinpath('dialogs/')

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess.py')

    opts.preprocess_opts(parser)
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    return opt


def build_save_dataset(corpus_type, fields, opt, save=True):
    assert corpus_type in ['train', 'valid', 'test']

    """
    Process the text corpus into example_dict iterator.
    """
    if corpus_type == 'train':
        corpus_file = opt.train_data
    elif corpus_type == 'test':
        corpus_file = opt.test_data
    elif corpus_type == 'valid':
        corpus_file = opt.valid_data
    conversations = mtdg.text_dataset.read_ubuntu_convs(corpus_file,
                                                       min_turn=opt.min_turn_length, max_turn=opt.max_turn_length,
                                                       min_seq=opt.min_seq_length, max_seq=opt.max_seq_length,
                                                       n_workers=opt.n_workers)
    # elif opt.data == "dailydialog":
    #     if corpus_type == 'train':
    #         corpus_file = opt.train_data
    #     elif corpus_type == 'valid':
    #         corpus_file = opt.valid_data
    #     else:
    #         corpus_file = opt.test_data
    #     conversations = mtdg.data.read_dailydialog_file(corpus_file, opt.max_turns, opt.max_seq_length)

     # = mtdg.data.read_dailydialog_file(tgt_corpus, opt.max_turns, opt.tgt_seq_length, "tgt")
    dataset = mtdg.data.Dataset(conversations, fields)

    if save:
        dataset.fields = []
        print("{:s}.{:s}.pt".format(opt.save_data, corpus_type))
        torch.save(dataset, "{:s}.{:s}.pt".format(opt.save_data, corpus_type))

    return dataset


def build_save_vocab(train_dataset, fields, opt, save=True):
    # We've empty'ed each dataset's `fields` attribute
    # when saving datasets, so restore them.
    train_dataset.fields = fields

    fields["conversation"].build_vocab(train_dataset, max_size=opt.vocab_size,
                              min_freq=opt.words_min_frequency)

    if save:
        # Can't save fields, so remove/reconstruct at training time.
        torch.save(mtdg.data.save_fields_to_vocab(fields), opt.save_data + '.vocab.pt')

    return fields



def main():
    opt = parse_args()

    print('Preparing for training ...')
    fields = mtdg.text_dataset.get_fields(opt)

    print("Building & saving training data...")
    train_dataset = build_save_dataset('train', fields, opt)

    print("Building & saving vocabulary...")
    fields = build_save_vocab(train_dataset, fields, opt)

    print("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)

    print("Building & saving test data...")
    build_save_dataset("test", fields, opt)

if __name__ == "__main__":
    main()