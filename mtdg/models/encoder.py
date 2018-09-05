import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

from mtdg.data import PAD_TOKEN
from mtdg.utils.convert import reverse_order_valid
from mtdg.models.cnn_factory import shape_transform, StackedCNN
from mtdg.utils.misc import aeq

import copy

class BaseRNNEncoder(nn.Module):
    def __init__(self):
        """Base RNN Encoder Class"""
        super(BaseRNNEncoder, self).__init__()

    @property
    def use_lstm(self):
        if hasattr(self, 'rnn'):
            return isinstance(self.rnn, nn.LSTM)
        else:
            raise AttributeError('no rnn selected')

    def init_h(self, batch_size=None, hidden=None, type=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden

        if self.use_lstm:
            return (torch.zeros(self.num_layers*self.num_directions,
                                      batch_size,
                                      self.hidden_size),
                    torch.zeros(self.num_layers*self.num_directions,
                                      batch_size,
                                      self.hidden_size))
        else:
            # return torch.zeros(self.num_layers*self.num_directions,
            #                             batch_size,
            #                             self.hidden_size).cuda()
            return type.data.new(self.num_layers*self.num_directions,
                                        batch_size,
                                        self.hidden_size).zero_()

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTM)
        """
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size

        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def forward(self, inputs, input_length, hidden=None):
        raise NotImplementedError


class EncoderRNN(BaseRNNEncoder):
    def __init__(self, vocab_size, embedding_size, padding_idx,
                 hidden_size, rnn=nn.GRU, num_layers=1, bidirectional=False,
                 dropout=0.0, bias=True):
        """Sentence-level Encoder"""
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)

        self.rnn = rnn(input_size=embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        bias=bias,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, inputs, input_length, hidden=None):
        """
        Args:
            inputs (Variable, LongTensor): [num_setences, max_utterance_length]
            input_length (Variable, LongTensor): [num_setences]
        Return:
            outputs (Variable): [max_utterance_length, batch_size * max_turn, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len = inputs.size()

        # Sort in decreasing order of length for pack_padded_sequence()
        input_length_sorted, indices = input_length.sort(descending=True)

        input_length_sorted = input_length_sorted.tolist()

        # [num_setences, max_utterance_length]
        inputs_sorted = inputs.index_select(0, indices)

        # [num_setences, max_utterance_length, embedding_dim]
        embedded = self.embedding(inputs_sorted)

        # batch_first=True
        rnn_input = pack_padded_sequence(embedded, input_length_sorted, batch_first=True)

        hidden = self.init_h(batch_size, hidden=hidden, type=embedded)

        # outputs: [num_setences, max_utterance_length, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, num_setences, hidden_size]
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)
        outputs, outputs_lengths = pad_packed_sequence(outputs, batch_first=True)

        # Reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                        hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        return outputs, hidden

class ContextRNN(BaseRNNEncoder):
    def __init__(self, input_size, context_size, rnn=nn.GRU, num_layers=1, dropout=0.0,
                 bidirectional=False, bias=True, batch_first=True):
        """Context-level Encoder"""
        super(ContextRNN, self).__init__()

        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = self.context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.rnn = rnn(input_size=input_size,
                        hidden_size=context_size,
                        num_layers=num_layers,
                        bias=bias,
                        batch_first=batch_first,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, encoder_hidden, conversation_turns, hidden=None):
        """
        Args:
            encoder_hidden (FloatTensor): [max_turns, batch_size, num_layers * direction * hidden_size]
            conversation_turns (LongTensor): [batch_size]
        Return:
            outputs (Variable): [batch_size, max_turns, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len , _  = encoder_hidden.size()

        # Sort for PackedSequence
        conv_turn_sorted, indices = conversation_turns.sort(descending=True)
        conv_turn_sorted = conv_turn_sorted.data.tolist()
        encoder_hidden_sorted = encoder_hidden.index_select(0, indices)

        rnn_input = pack_padded_sequence(encoder_hidden_sorted, conv_turn_sorted, batch_first=True)

        hidden = self.init_h(batch_size, hidden=hidden, type=encoder_hidden)

        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)

        # outputs: [batch_size, max_turns, context_size]
        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)

        # reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                    hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        # outputs: [batch, max_turns, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        return outputs, hidden

    def step(self, encoder_hidden, hidden):

        batch_size = encoder_hidden.size(0)
        # encoder_hidden: [1, batch_size, hidden_size]
        encoder_hidden = torch.unsqueeze(encoder_hidden, 1)

        if hidden is None:
            hidden = self.init_h(batch_size, hidden=None)

        outputs, hidden = self.rnn(encoder_hidden, hidden)
        return outputs, hidden


# SCALE_WEIGHT = 0.5 ** 0.5


# Stack Gated Convolutional Neural Network
class CNNEncoder(nn.Module):
    """
    Encoder built on CNN based on
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """
    def __init__(self, vocab_size, embedding_size, padding_idx,
                 num_layers, cnn_kernel_size, cnn_kernel_width, dropout,
                 topic_num=None, topic_key_size=None, topic_value_size=None, concat=True):
        super(CNNEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)

        self.linear = nn.Linear(embedding_size, cnn_kernel_size)
        self.cnn = StackedCNN(num_layers, cnn_kernel_size,
                              cnn_kernel_width, dropout)

        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()

        assert cnn_kernel_size == topic_key_size, "The cnn_kernel_size and topic_size should be equal."
        if topic_num is not None:
            self.topic_keys = nn.Parameter(torch.Tensor(topic_num, topic_key_size), requires_grad=True)
            self.topic_values = nn.Parameter(torch.Tensor(topic_num, topic_value_size), requires_grad=True)
            # self.topic_values = self.topic_keys
        self.concat = concat


    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`onmt.modules.EncoderBase.forward()`"""
        # input = input.t_().contiguous()

        emb = self.embedding(input)
        # batch, s_len, emb_dim = emb.size()

        # emb = emb.transpose(0, 1).contiguous()
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap).squeeze(3)

        out = self.maxpooling(out).squeeze(2)
        features = self.relu(out)

        topic_dist = F.softmax(torch.mm(out, self.topic_keys.t()), dim=1)  # (batch_size, topic_num)
        topic_aware_representation = topic_dist.mm(self.topic_values)

        if self.concat:
            out = torch.cat((features, topic_aware_representation), 1)
            return topic_aware_representation, out.unsqueeze(0)     # (num_layers * direction, num_sentences, hidden_size)
        else:
            return topic_aware_representation, features.unsqueeze(0)
        # out = torch.cat((out, topic_aware_representation), 1)
        # return topic_aware_representation, out.unsqueeze(0).transpose(0, 1).contiguous()


# Basic CNN
class BaseCNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx,
                 num_layers, cnn_kernel_size, cnn_kernel_width, dropout,
                 topic_num=None, topic_key_size=None, topic_value_size=None):
        super(CNNEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)

        self.linear = nn.Linear(embedding_size, cnn_kernel_size)
        self.cnn = StackedCNN(num_layers, cnn_kernel_size,
                              cnn_kernel_width, dropout)

        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()

        assert cnn_kernel_size == topic_key_size, "The cnn_kernel_size and topic_size should be equal."
        if topic_num is not None:
            self.topic_keys = nn.Parameter(torch.Tensor(topic_num, topic_key_size), requires_grad=True)
            self.topic_values = nn.Parameter(torch.Tensor(topic_num, topic_value_size), requires_grad=True)
            # self.topic_values = self.topic_keys


    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`onmt.modules.EncoderBase.forward()`"""
        # input = input.t_().contiguous()

        emb = self.embedding(input)
        # batch, s_len, emb_dim = emb.size()

        # emb = emb.transpose(0, 1).contiguous()
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap).squeeze(3)

        out = self.maxpooling(out).squeeze(2)
        out = self.relu(out)

        topic_dist = F.softmax(torch.mm(out, self.topic_keys.t()), dim=1)  # (batch_size, topic_num)
        topic_aware_representation = topic_dist.mm(self.topic_values)

        out = torch.cat((out, topic_aware_representation), 1)

        return topic_aware_representation, out.unsqueeze(0).transpose(0, 1).contiguous()


class CNNBase(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx,
                 num_layers, cnn_kernel_size, cnn_kernel_width, dropout,
                 topic_num=None, topic_key_size=None, topic_value_size=None, concat=True):
        super(CNNBase, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)

        self.cnn_kernel_width = cnn_kernel_width
        self.output_size = len(self.cnn_kernel_width) * cnn_kernel_size
        self.convs = nn.ModuleList([nn.Conv2d(1, cnn_kernel_size, (w, embedding_size)) for w in self.cnn_kernel_width])

        # self.bn = nn.BatchNorm2d(opt.cnn_kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

        assert len(self.cnn_kernel_width) * cnn_kernel_size == topic_key_size, "The cnn_kernel_size and topic_size should be equal."
        if topic_num is not None:
            self.topic_keys = nn.Parameter(torch.Tensor(topic_num, topic_key_size), requires_grad=True)
            self.topic_values = nn.Parameter(torch.Tensor(topic_num, topic_value_size), requires_grad=True)

        self.concat = concat

    def forward(self, input, init_state=None):
        embedding = self.embedding(input)
        # emb_reshape = shape_transform(embedding)
        x = [F.relu(conv(embedding.unsqueeze(1))).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        # x = [self.relu(self.bn(conv(embeddings.transpose(0, 1).unsqueeze(1)).squeeze(3))) for conv in self.convs]
        x = torch.cat(x, 1)
        features = self.dropout(x)

        topic_dist = F.softmax(torch.mm(features, self.topic_keys.t()), dim=1)  # (batch_size, topic_num)
        topic_aware_representation = topic_dist.mm(self.topic_values)

        if self.concat:
            out = torch.cat((features, topic_aware_representation), 1)
            return topic_aware_representation, out.unsqueeze(0)     # (num_layers * direction, num_sentences, hidden_size)
        else:
            return topic_aware_representation, features.unsqueeze(0)

class BiGRUEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, topic_num=20, topic_key_size=50, topic_value_size=50):
        super(BiGRUEncoder, self).__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # 2. Bi-directional LSTM
        self.hidden_size = hidden_size
        self.biGRU = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=True,
        )
        self.max_over_time = nn.AdaptiveMaxPool1d(1)

        if topic_num is not None:
            self.topic_keys = nn.Parameter(torch.Tensor(topic_num, topic_key_size), requires_grad=True)
            self.topic_values = nn.Parameter(torch.Tensor(topic_num, topic_value_size), requires_grad=True)

    def forward(self, input, init_state=None):
        embedding = self.embedding(input)
        # emb_reshape = shape_transform(embedding)
        x = [F.relu(conv(embedding.unsqueeze(1))).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        # x = [self.relu(self.bn(conv(embeddings.transpose(0, 1).unsqueeze(1)).squeeze(3))) for conv in self.convs]
        x = torch.cat(x, 1)
        features = self.dropout(x)

        topic_dist = F.softmax(torch.mm(features, self.topic_keys.t()), dim=1)  # (batch_size, topic_num)
        topic_aware_representation = topic_dist.mm(self.topic_values)

        out = torch.cat((features, topic_aware_representation), 1)

        return topic_aware_representation, out.unsqueeze(0)     # (num_layers * direction, num_sentences, hidden_size)


class TopicDriftRNN(BaseRNNEncoder):
    def __init__(self, input_size, context_size, rnn=nn.GRU, num_layers=1, dropout=0.0,
                 bidirectional=False, bias=True, batch_first=True):
        """Topic-level Encoder"""
        super(TopicDriftRNN, self).__init__()

        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = self.context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.rnn = rnn(input_size=input_size,
                        hidden_size=context_size,
                        num_layers=num_layers,
                        bias=bias,
                        batch_first=batch_first,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, encoder_hidden, conversation_turns, hidden=None):
        """
        Args:
            encoder_hidden (FloatTensor): [max_turns, batch_size, num_layers * direction * hidden_size]
            conversation_turns (LongTensor): [batch_size]
        Return:
            outputs (Variable): [batch_size, max_turns, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len , _  = encoder_hidden.size()

        # Sort for PackedSequence
        conv_turn_sorted, indices = conversation_turns.sort(descending=True)
        conv_turn_sorted = conv_turn_sorted.data.tolist()
        encoder_hidden_sorted = encoder_hidden.index_select(0, indices)

        rnn_input = pack_padded_sequence(encoder_hidden_sorted, conv_turn_sorted, batch_first=True)

        hidden = self.init_h(batch_size, hidden=hidden, type=encoder_hidden)

        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)

        # outputs: [batch_size, max_turns, context_size]
        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)

        # reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                    hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        # outputs: [batch, max_turns, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        return outputs, hidden

    def step(self, encoder_hidden, hidden):

        batch_size = encoder_hidden.size(0)
        # encoder_hidden: [1, batch_size, hidden_size]
        encoder_hidden = torch.unsqueeze(encoder_hidden, 1)

        if hidden is None:
            hidden = self.init_h(batch_size, hidden=None)

        outputs, hidden = self.rnn(encoder_hidden, hidden)
        return outputs, hidden



class TopicEncoderRNN(BaseRNNEncoder):
    def __init__(self, vocab_size, embedding_size, padding_idx,
                 hidden_size, rnn=nn.GRU, num_layers=1, bidirectional=False,
                 dropout=0.0, bias=True,
                 topic_num=None, topic_key_size=None, topic_value_size=None, concat=True):
        """Sentence-level Encoder"""
        super(TopicEncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)

        self.rnn = rnn(input_size=embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        bias=bias,
                        dropout=dropout,
                        bidirectional=bidirectional)

        if topic_num is not None:
            self.topic_keys = nn.Parameter(torch.Tensor(topic_num, topic_key_size), requires_grad=True)
            self.topic_values = nn.Parameter(torch.Tensor(topic_num, topic_value_size), requires_grad=True)

        self.self_attention_gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.self_attention_linear = nn.Linear(hidden_size, topic_key_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.concat = concat

    def forward(self, inputs, input_length, hidden=None):
        """
        Args:
            inputs (Variable, LongTensor): [num_setences, max_utterance_length]
            input_length (Variable, LongTensor): [num_setences]
        Return:
            outputs (Variable): [max_utterance_length, batch_size * max_turn, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len = inputs.size()

        # Sort in decreasing order of length for pack_padded_sequence()
        input_length_sorted, indices = input_length.sort(descending=True)

        input_length_sorted = input_length_sorted.tolist()

        # [num_setences, max_utterance_length]
        inputs_sorted = inputs.index_select(0, indices)

        # [num_setences, max_utterance_length, embedding_dim]
        embedded = self.embedding(inputs_sorted)

        # batch_first=True
        rnn_input = pack_padded_sequence(embedded, input_length_sorted, batch_first=True)

        hidden = self.init_h(batch_size, hidden=hidden, type=embedded)

        # outputs: [num_setences, max_utterance_length, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, num_setences, hidden_size]
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)
        outputs, outputs_lengths = pad_packed_sequence(outputs, batch_first=True)

        # Reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                        hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        out = torch.bmm(
            self.self_attention_gate(outputs.view(-1, outputs.size(2))).view(outputs.size(0), 1, outputs.size(1)),
            self.self_attention_linear(outputs.view(-1, outputs.size(2))).view(outputs.size(0), outputs.size(1), -1)
        ).squeeze(1)
        out = self.dropout(self.relu(out))

        topic_dist = F.softmax(torch.mm(out, self.topic_keys.t()), dim=1)  # (batch_size, topic_num)
        topic_hidden = topic_dist.mm(self.topic_values)

        return topic_hidden, hidden