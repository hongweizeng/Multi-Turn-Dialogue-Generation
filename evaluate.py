import os
import math
import subprocess
import argparse
import codecs
import string

from gensim.models import KeyedVectors
import numpy as np

import torch

import mtdg
import mtdg.opts as opts
from mtdg.utils.misc import use_gpu
from mtdg.utils.logging import logger

from mtdg.model_builder import build_model, load_test_model
from mtdg.trainer import build_trainer

from tools.test_rouge import test_rouge, rouge_results_to_str
from tools.embedding_metrics import average, greedy_match, extrema_score

def report_ground_truth(opt):
    assert opt.ckpt is not None and opt.data is not None

    # 1. Model & Fields
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = load_test_model(opt, dummy_opt.__dict__)

    # 2. Dataset & Iterator
    if ".pt" in opt.data:
        dataset = torch.load(opt.data)
        dataset.fields = fields
    else:
        conversations = mtdg.data.read_dailydialog_file(opt.data)
        dataset = mtdg.data.Dataset(conversations, fields)
    device = "cuda" if use_gpu(opt) else "cpu"
    data_iter = mtdg.data.OrderedIterator(dataset=dataset, batch_size=opt.batch_size, device=device,
                                          train=False, sort=False, repeat=False, shuffle=False)
    # 3. Trainer
    trainer = build_trainer(model_opt, model, fields, optim=None, device=device)

    # 4. Run on test data
    test_stats = trainer.valid(data_iter)
    if opt.report_ppl:
        msg = ("Perplexity: %g" % test_stats.ppl())
        logger.info(msg)
    if opt.report_xent:
        msg = ("Xent: %g" % test_stats.xent())
        logger.info(msg)
    if opt.report_accuracy:
        msg = ("Accuracy: %g" % test_stats.accuracy())
        logger.info(msg)


def report_bleu(tgt_path, out_path):
    output_file = codecs.open(out_path, 'r', 'utf-8')
    res = subprocess.check_output("perl tools/multi-bleu.perl %s"
                                  % (tgt_path),
                                  stdin=output_file,
                                  shell=True).decode("utf-8")

    msg = res.strip()
    logger.info(msg)


def report_rouge(tgt_path, out_path):
    tgt_file = codecs.open(tgt_path, "r", "utf-8")
    out_file = codecs.open(out_path, "r", "utf-8")
    results_dict = test_rouge(out_file, tgt_file)
    msg = rouge_results_to_str(results_dict)
    logger.info(msg)


def report_distinct(tgt_path):
    distinct_1 = set()
    distinct_2 = set()

    lines = codecs.open(tgt_path, "r", "utf-8").readlines()

    total_words = 0
    for line in lines:
        # words = [word for word in line.split() if word not in string.punctuation]
        words = [word for word in line.split()]
        distinct_1.update(words)
        distinct_2.update([words[i] + words[i + 1] for i in range(len(words) - 2)])
        total_words += len(words)
    # print("distinct-1 = ", len(distinct_1) * 1.0 / total_words, "distinct number = ", len(distinct_1))
    # print("distinct-2 = ", len(distinct_2) * 1.0 / total_words, "distinct number = ", len(distinct_2))
    msg = ("Distinct-1 = %.4f, Distinct-2 = %.4f" % (
        len(distinct_1) * 1.0 / total_words,  len(distinct_2) * 1.0 / total_words))
    logger.info(msg)


def report_embedding(tgt_path, out_path, embedding_path):
    w2v = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

    r = average(tgt_path, out_path, w2v)
    msg = ("Embedding Average Score: %f +/- %f ( %f )" %(r[0], r[1], r[2]))
    logger.info(msg)

    r = greedy_match(tgt_path, out_path, w2v)
    msg = ("Greedy Matching Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))
    logger.info(msg)

    r = extrema_score(tgt_path, out_path, w2v)
    msg = ("Extrema Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))
    logger.info(msg)



def main(opt):
    if opt.report_ppl or opt.report_xent or opt.report_accuracy:
        report_ground_truth(opt)
    if opt.report_bleu:
        report_bleu(opt.target, opt.output)
    if opt.report_rouge:
        report_rouge(opt.target, opt.output)
    if opt.report_distinct:
        report_distinct(opt.output)
    if opt.report_embedding:
        assert os.path.exists(opt.embeddings)
        report_embedding(opt.target, opt.output, opt.embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train.py')
    opts.generate_opts(parser)
    opts.evaluate_opts(parser)
    opt = parser.parse_args()
    main(opt)