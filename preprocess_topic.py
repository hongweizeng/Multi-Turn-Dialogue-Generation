from __future__ import division

import re
import math
import torch
import torchtext

import os
import argparse
from pathlib import Path
import sys

import mtdg.opts as opts
from mtdg.inputters.topic_dataset import TopicDataset, TopicField, save_fields_to_vocab, load_fields_from_vocab, get_fields


# def main():
#
#     print("Loading lines")
#     # lines = loadLines(cornell_dir.joinpath("movie_lines.txt"))
#     truncted_lines = []
#     # writer = open("data/dot/valid.topic.txt", "w")
#     # lines = open("data/ptb/ptb.valid.txt", 'r').readlines()
#     writer = open("data/train/train.topic.txt", "w")
#     lines = open("data/train/dialogues_train.txt", 'r').readlines()
#     for line in lines:
#         sentences= line.rstrip().split("__eou__")
#         for sentence in sentences:
#             words = sentence.rstrip().split()
#             truncted_lines.append(words)
#
#     from collections import Counter
#
#     counter = Counter()
#     for words in truncted_lines:
#         counter.update(words)
#
#     stopwords = set([item.strip().lower() for item in open("tools/stopwords.txt").readlines()])
#     freqwords = set([item[0] for item in sorted(counter.most_common(int(float(len(counter) * 0.001))))])
#     alpha_check = re.compile("[a-zA-Z]")
#     symbols = set([w for w in counter if  ((alpha_check.search(w) == None) or w.startswith("'")) ])
#     ignore = stopwords | freqwords | symbols | set(["n't"])
#
#
#     for words in truncted_lines:
#         non_stop_words = [w for w in words if w not in ignore]
#         # for word in words:
#         #     if word not in stopwords:
#         #         non_stop_words.append(word)
#         # if len(words) > 5:
#         #     topic_words = sorted(words, key=lambda d: counter[d])[:5]
#         #     for word in topic_words:
#
#         for i in range(math.ceil(len(non_stop_words) / 3)):
#             seq = non_stop_words[i*3: i*3+4]
#             writer.write(" ".join(words) + " +++$+++ " + " ".join(seq) + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess_topic.py')

    opts.preprocess_opts(parser)
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    return opt


def main():
    opt = parse_args()

    print("Building & saving training data...")
    fields = get_fields()
    train_dataset = TopicDataset(opt.save_data + ".train.pt", fields)
    fields = train_dataset.fields
    train_dataset.fields = []
    torch.save(train_dataset, "{:s}.{:s}.topic".format(opt.save_data, "train"))

    # print("Building & saving vocabulary...")
    # train_dataset.fields = fields
    # fields["text"].build_vocab(train_dataset, max_size=20000, min_freq=2)
    # fields["target"].vocab = fields["text"].vocab
    # torch.save(save_fields_to_vocab(fields), "{:s}.{:s}.topic".format(opt.save_data, "vocab"))

    print("Building & saving validation data...")
    valid_dataset = TopicDataset(opt.save_data + ".valid.pt", fields)
    valid_dataset.fields = []
    torch.save(valid_dataset, "{:s}.{:s}.topic".format(opt.save_data, "valid"))


if __name__ == '__main__':
    main()