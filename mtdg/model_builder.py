"""
This file is for models creation.
"""
from mtdg.data import PAD_TOKEN, load_fields_from_vocab
from mtdg.models.encoder import EncoderRNN, ContextRNN, CNNEncoder, CNNBase, BiGRUEncoder, TopicDriftRNN, TopicEncoderRNN
from mtdg.models.feedforward import FeedForward
from mtdg.models.decoder import DecoderRNN, TopicGatedDecoder
from mtdg.models.rnn_factory import StackedGRUCell, StackedLSTMCell
from mtdg.models.gate import BothContextGate
from mtdg.models.models import HRED, TDCM, TDACM

from mtdg.utils.logging import logger
from mtdg.utils.misc import use_gpu

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn import LSTM, GRU


import os

def build_HRED(opt, fields):
    vocab = fields["conversation"].vocab

    encoder = EncoderRNN(len(vocab), opt.src_word_vec_size, padding_idx=vocab.stoi[PAD_TOKEN], hidden_size=opt.enc_rnn_size,
                         rnn=GRU if opt.enc_rnn_type == "GRU" else LSTM, num_layers=opt.enc_layer,
                         bidirectional=opt.bidirectional, dropout=opt.dropout)
    context_input_size = opt.enc_layer * opt.enc_rnn_size
    if opt.bidirectional:
        context_input_size *= 2
    context_encoder = ContextRNN(context_input_size, opt.context_rnn_size,
                                 rnn=GRU if opt.context_rnn_type == "GRU" else LSTM,
                                 num_layers=opt.context_layer,
                                 dropout=opt.dropout)
    context2decoder = FeedForward(opt.context_rnn_size, opt.dec_layer * opt.dec_rnn_size)
    decoder = DecoderRNN(vocab, len(vocab), opt.tgt_word_vec_size, hidden_size=opt.dec_rnn_size,
                         rnncell=StackedGRUCell if opt.dec_rnncell_type == "GRU" else StackedLSTMCell,
                         num_layers=opt.dec_layer, dropout=opt.dropout)

    model = HRED(encoder, context_encoder, context2decoder, decoder)

    return model


def build_TDCM(opt, fields):
    vocab = fields["conversation"].vocab
    if opt.enc_cnn_type == "base":
        encoder = CNNBase(len(vocab), opt.word_vec_size, vocab.stoi[PAD_TOKEN],
                             opt.enc_layer,  opt.cnn_kernel_size, opt.cnn_kernel_width, opt.dropout,
                             opt.topic_num, opt.topic_key_size, opt.topic_value_size)
        context_input_size = len(opt.cnn_kernel_width) * opt.cnn_kernel_size + opt.topic_value_size
    elif opt.enc_cnn_type == "gate":
        encoder = CNNEncoder(len(vocab), opt.word_vec_size, vocab.stoi[PAD_TOKEN],
                             opt.enc_layer,  opt.cnn_kernel_size, opt.cnn_kernel_width, opt.dropout,
                             opt.topic_num, opt.topic_key_size, opt.topic_value_size)
        context_input_size = opt.cnn_kernel_size + opt.topic_value_size
    elif opt.enc_cnn_type == "gru":
        encoder = BiGRU()
        context_input_size = opt.cnn_kernel_size + opt.topic_value_size
    else:
        raise NotImplementedError
    # context_input_size = opt.cnn_kernel_size + opt.topic_key_size
    context_encoder = ContextRNN(context_input_size, opt.context_rnn_size,
                                 rnn=GRU if opt.context_rnn_type == "GRU" else LSTM,
                                 num_layers=opt.context_layer,
                                 dropout=opt.dropout)
    context2decoder = FeedForward(opt.context_rnn_size, opt.dec_layer * opt.dec_rnn_size)
    # decoder = DecoderRNN(vocab, len(vocab), opt.tgt_word_vec_size, hidden_size=opt.dec_rnn_size,
    #                      rnncell=StackedGRUCell if opt.dec_rnncell_type == "GRU" else StackedLSTMCell,
    #                      num_layers=opt.dec_layer, dropout=opt.dropout)

    topic_gate = None if not opt.topic_gate else \
    BothContextGate(opt.word_vec_size, opt.dec_rnn_size, opt.topic_value_size, opt.dec_rnn_size)

    decoder = TopicGatedDecoder(vocab, len(vocab), opt.tgt_word_vec_size, hidden_size=opt.dec_rnn_size,
                                rnncell=StackedGRUCell if opt.dec_rnncell_type == "GRU" else StackedLSTMCell,
                                num_layers=opt.dec_layer, dropout=opt.dropout,
                                topic_gate=topic_gate)

    model = TDCM(encoder, context_encoder, context2decoder, decoder)

    predictor = nn.Sequential(
        nn.Linear(context_input_size, len(vocab)),
        nn.Softmax()
    )
    model.predictor = predictor
    return model


def build_TDACM(opt, fields):
    vocab = fields["conversation"].vocab
    if opt.enc_cnn_type == "base":
        encoder = CNNBase(len(vocab), opt.word_vec_size, vocab.stoi[PAD_TOKEN],
                          opt.enc_layer, opt.cnn_kernel_size, opt.cnn_kernel_width, opt.dropout,
                          opt.topic_num, opt.topic_key_size, opt.topic_value_size, concat=False)
        context_input_size = len(opt.cnn_kernel_width) * opt.cnn_kernel_size# + opt.topic_value_size
    elif opt.enc_cnn_type == "gate":
        _cnn_kernel_width = 3
        encoder = CNNEncoder(len(vocab), opt.word_vec_size, vocab.stoi[PAD_TOKEN],
                             opt.enc_layer, opt.cnn_kernel_size, _cnn_kernel_width, opt.dropout,
                             opt.topic_num, opt.topic_key_size, opt.topic_value_size, concat=False)
        context_input_size = opt.cnn_kernel_size# + opt.topic_value_size
    elif opt.enc_cnn_type == "rnn":
        encoder = TopicEncoderRNN(len(vocab), opt.src_word_vec_size, padding_idx=vocab.stoi[PAD_TOKEN], hidden_size=opt.enc_rnn_size,
                   rnn=GRU if opt.enc_rnn_type == "GRU" else LSTM, num_layers=opt.enc_layer,
                   bidirectional=opt.bidirectional, dropout=opt.dropout,
                   topic_num=opt.topic_num, topic_key_size=opt.topic_key_size, topic_value_size=opt.topic_value_size)
    #     encoder = BiGRU()
        context_input_size = opt.enc_rnn_size
    else:
        raise NotImplementedError
    context_encoder = ContextRNN(context_input_size, opt.context_rnn_size,
                                 rnn=GRU if opt.context_rnn_type == "GRU" else LSTM,
                                 num_layers=opt.context_layer,
                                 dropout=opt.dropout)
    context2decoder = FeedForward(opt.context_rnn_size, opt.dec_layer * opt.dec_rnn_size)

    topic_encoder = TopicDriftRNN(opt.topic_value_size, opt.topic_rnn_size,
                                        rnn=GRU if opt.topic_rnn_type == "GRU" else LSTM,
                                        num_layers=opt.topic_layer,
                                        dropout=opt.dropout)
    topic2decoder = FeedForward(opt.topic_rnn_size, opt.dec_layer * opt.dec_rnn_size)
    topic_state_size = opt.dec_layer * opt.dec_rnn_size

    topic_gate = None if not opt.topic_gate else \
        BothContextGate(opt.word_vec_size, opt.dec_rnn_size, topic_state_size, opt.dec_rnn_size)

    decoder = TopicGatedDecoder(vocab, len(vocab), opt.tgt_word_vec_size, hidden_size=opt.dec_rnn_size,
                                rnncell=StackedGRUCell if opt.dec_rnncell_type == "GRU" else StackedLSTMCell,
                                num_layers=opt.dec_layer, dropout=opt.dropout,
                                topic_gate=topic_gate)

    model = TDACM(encoder, context_encoder, topic_encoder, context2decoder, topic2decoder, decoder)

    predictor = nn.Sequential(
        nn.Linear(context_input_size, len(vocab)),
        nn.Softmax()
    )
    model.predictor = predictor
    return model

def build_model(model_opt, fields, gpu, checkpoint):
    """ Build the Model """
    logger.info('Building model...')
    if model_opt.model == "HRED":
        model = build_HRED(model_opt, fields)
    elif model_opt.model == "TDCM":
        model = build_TDCM(model_opt, fields)
    elif model_opt.model == "TDACM":
        model = build_TDACM(model_opt, fields)
    else:
        raise NotImplementedError

    device = torch.device("cuda" if gpu else "cpu")

    if model_opt.share_embeddings:
        model.decoder.embedding.weight = model.encoder.embedding.weight

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        # topic orthogonal
        if model_opt.model == "TDACM" or model_opt.model == "TDCM":
            nn.init.orthogonal_(model.encoder.topic_keys)
            nn.init.orthogonal_(model.encoder.topic_values)

        vocab = fields["conversation"].vocab
        embedding_pt_path = model_opt.data + ".embedding.pt"
        if not os.path.exists(embedding_pt_path):
            from gensim.models import KeyedVectors
            w2v = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)
            for token in vocab.itos:
                if token in w2v:
                    model.encoder.embedding.weight.data[vocab.stoi[token]] = torch.from_numpy(w2v[token])
            torch.save(model.encoder.embedding.weight.data, embedding_pt_path)
        else:
            model.encoder.embedding.weight.data.copy_(torch.load(embedding_pt_path))

    model.to(device)
    logger.info(model)
    return model

def load_test_model(opt, dummy_opt):
    """ Load model for Inference """
    checkpoint = torch.load(opt.ckpt, map_location=lambda storage, loc: storage)
    fields = load_fields_from_vocab(checkpoint['vocab'])

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    model = build_model(model_opt, fields, use_gpu(opt), checkpoint)
    model.eval()
    return fields, model, model_opt