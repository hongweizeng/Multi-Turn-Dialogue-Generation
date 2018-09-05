#!/usr/bin/env python

from __future__ import division

import argparse
import os
import sys
import random
from collections import deque

import torch
import torch.nn as nn
from torch import cuda
import visdom

import mtdg.opts as opts
import mtdg.utils
from mtdg.utils.misc import use_gpu
from mtdg.data import  build_iterator, _load_fields
from mtdg.trainer import build_trainer
from mtdg.model_builder import build_model
from mtdg.model_saver import build_model_saver
from mtdg.utils.logging import logger
from mtdg.utils.optimizers import build_optim


def training_opt_postprocessing(opt):
    if torch.cuda.is_available() and not opt.gpuid:
        logger.info("WARNING: You have a CUDA device, should run with -gpuid")

    if opt.gpuid:
        torch.cuda.set_device(opt.gpuid[0])
        if opt.seed > 0:
            # this one is needed for torchtext random call (shuffled iterator)
            # in multi gpu it ensures datasets are read in the same order
            random.seed(opt.seed)
            # These ensure same initialization in multi gpu mode
            torch.manual_seed(opt.seed)
            torch.cuda.manual_seed(opt.seed)

    return opt



def main(opt):
    opt = training_opt_postprocessing(opt)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        start_epoch = checkpoint["epoch"] + 1
    else:
        checkpoint = None
        model_opt = opt
        start_epoch = 0


    # Load fields generated from preprocess phase.
    train_dataset = torch.load(opt.data + '.valid.pt')
    fields = _load_fields(train_dataset, opt, checkpoint)

    train_iter = build_iterator("train", fields, opt)
    # train_iter = build_iterator("valid", fields, opt)
    valid_iter = build_iterator("valid", fields, opt, is_train=False)
    # test_iter = build_iterator("test", fields, opt, is_train=False)


    # Build model.
    model = build_model(model_opt, fields, use_gpu(opt), checkpoint)

    # Build optimizer.
    # optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model, opt, fields, optim)

    # Build trainer
    device = "cuda" if use_gpu(opt) else "cpu"
    # if opt.vis_logger:
    #     vis_logger = visdom.Visdom(
    #         server='http://202.117.54.73',
    #         endpoint='events',
    #         port=8097,
    #         ipv6=True,
    #         # proxy=None,
    #         env='multi_turn_dialog')
    trainer = build_trainer(
        # opt, model, fields, optim, device=device, model_saver=model_saver, vis_logger=vis_logger)
        opt, model, fields, optim, device=device, model_saver=model_saver, vis_logger=None)

    # 6. Do training.
    if opt.model == "TDCM" or opt.model == "TDACM":
        # 6.1 Data
        train_topic_iter = build_iterator("train", fields, opt, is_topic=True)
        valid_topic_iter = build_iterator("valid", fields, opt, is_train=False, is_topic=True)
        # test_topic_iter = build_iterator("test", fields, opt, is_train=False, is_topic=True)

        # optimizer = build_optim(model, opt, checkpoint, is_topic=True)
        topic_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        topic_criterion = mtdg.TopicLossCompute(fields["conversation"].vocab).to(device)
        # trainer.train_topic(train_topic_iter, valid_topic_iter, 10, 1, topic_criterion, topic_optimizer, test_iter=None)
        # # checkpoint = torch.load(model_saver.best_checkpoint)
        # checkpoint = torch.load(model_saver.checkpoint_queue[-1])
        # model.load_state_dict(checkpoint["model"])
        # model_saver._rm_checkpoint(model_saver.best_checkpoint)
        # model_saver.reset()
        trainer.train(train_iter, valid_iter, [start_epoch, opt.epochs], 1,
                      train_topic_iter=train_topic_iter, valid_topic_iter=valid_topic_iter,
                      topic_criterion=topic_criterion, topic_optimizer=topic_optimizer)
    else:
        trainer.train(train_iter, valid_iter, [start_epoch, opt.epochs], 1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train.py')
    opts.train_opts(parser)
    opts.model_opts(parser)
    opt = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpuid[0])

    main(opt)