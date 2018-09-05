#!/usr/bin/env python
""" Generator Class and builder """
from __future__ import print_function
import argparse
import codecs
import os
import tqdm
from pathlib import Path

import torch

from itertools import count

import mtdg.model_builder
import mtdg.data
import mtdg.opts as opts
from mtdg.utils.misc import use_gpu
from mtdg.data import EOS_TOKEN, BOS_TOKEN, PAD_TOKEN


def build_generator(opt):
    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = mtdg.model_builder.load_test_model(opt, dummy_opt.__dict__)

    generator = Generator(model, fields, output_file=opt.output, target_file=opt.target, cuda=use_gpu(opt))
    return generator


class Generator(object):
    def __init__(self, model, fields, output_file, target_file=None, cuda=False):
        self.model = model
        self.fields = fields
        self.target_writer = open(target_file, "w")
        self.output_writer = open(output_file, "w")
        self.cuda = cuda

    def generate(self, data_path=None, data_iter=None, batch_size=None):

        assert data_path is not None or data_iter is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")

        if ".pt" in data_path:
            # dialogs_dir = Path("data/ubuntu/dialogs")
            # corpus_file = Path("data/ubuntu/meta/testfiles.csv")
            # # corpus_file = ubuntu_meta_dir.joinpath('testfiles.csv')
            # conversations = mtdg.text_dataset.read_ubuntu_file(corpus_file, dialogs_dir,
            #                                                    min_turn=opt.min_turn_length,
            #                                                    max_turn=opt.max_turn_length,
            #                                                    min_seq=opt.min_seq_length, max_seq=opt.max_seq_length,
            #                                                    n_workers=opt.n_workers)
            dataset = torch.load(data_path)
            dataset.fields = self.fields
        else:
            conversations = mtdg.data.read_dailydialog_file(data_path)
        # = mtdg.data.read_dailydialog_file(tgt_corpus, opt.max_turns, opt.tgt_seq_length, "tgt")
            dataset = mtdg.data.Dataset(conversations, self.fields)

        device = "cuda" if self.cuda else "cpu"

        data_iter = mtdg.data.OrderedIterator(dataset=dataset, batch_size=batch_size, device=device,
                        train=False, sort=False, repeat=False, shuffle=False)

        for batch in data_iter:
            input_sentences, target_sentences = batch.conversation
            input_length, target_length = batch.length
            turn = batch.turn
            self.generate_sentence(input_sentences, input_length, turn, target_sentences)



    def generate_sentence(self, input_sentences, input_sentence_length,
                          input_conversation_length, target_sentences):
        self.model.eval()

        # [batch_size, max_seq_len, vocab_size]
        generated_sentences = self.model(
            input_sentences,
            input_sentence_length,
            input_conversation_length,
            target_sentences,
            decode=True)

        # write output to file
        for input_sent, target_sent, output_sent in zip(input_sentences, target_sentences, generated_sentences):
            # input_sent = self.decode(input_sent)
            target_sent = self.decode(target_sent)
            output_sent = '\n'.join([self.decode(sent) for sent in output_sent])
            # s = '\n'.join(['Input sentence: ' + input_sent,
            #                'Ground truth: ' + target_sent,
            #                'Generated response: ' + output_sent + '\n'])
            self.target_writer.write(target_sent + "\n")
            self.output_writer.write(output_sent + "\n")


    def id2sent(self, id_list):
        """list of id => list of tokens (Single sentence)"""
        vocab = self.fields["conversation"].vocab
        id_list = id_list.tolist()
        sentence = []
        for id in id_list:
            word = vocab.itos[id]
            if word not in [EOS_TOKEN, BOS_TOKEN, PAD_TOKEN]:
                sentence.append(word)
            if word == EOS_TOKEN:
                break
        return sentence

    def decode(self, id_list):
        sentence = self.id2sent(id_list)
        return ' '.join(sentence)