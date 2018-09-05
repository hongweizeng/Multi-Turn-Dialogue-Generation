from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


import mtdg
import mtdg.data
import mtdg.utils
from mtdg.utils.misc import sequence_mask, _sequence_mask
from mtdg.data import PAD_TOKEN


class LanguageLossCompute(nn.Module):
    """
        Class for managing efficient loss computation. Handles
        sharding next step predictions and accumulating mutiple
        loss computations


        Users can implement their own loss computation strategy by making
        subclass of this one.  Users need to implement the _compute_loss()
        and make_shard_state() methods.

        Args:
            tgt_vocab (:obj:`Vocab`) :
                 torchtext vocab object representing the target output
            normalzation (str): normalize by "sents" or "tokens"
        """

    def __init__(self, tgt_vocab, normalization="sents", label_smoothing=0.0):
        super(LanguageLossCompute, self).__init__()
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[PAD_TOKEN]
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def monolithic_compute_loss(self, scores, target):
        shard_state = self._make_shard_state(scores, target)
        _, batch_stats = self._compute_loss(**shard_state)
        return batch_stats

    def sharded_compute_loss(self, scores, target, shard_size):
        batch_size = target.size(0)
        # scores = F.log_softmax(self._bottle(scores), dim=-1)
        batch_stats = mtdg.utils.Statistics()
        shard_state = self._make_shard_state(scores, target)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(**shard)
            loss.div(batch_size).backward()
            batch_stats.update(stats)

        return batch_stats

    def _make_shard_state(self, scores, target):
        return {
            "scores": scores,
            "target": target,
        }

    def _compute_loss(self, scores, target):
        scores = F.log_softmax(self._bottle(scores), dim=-1)
        gtruth = target.view(-1)
        loss = self.criterion(scores, gtruth)
        loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        return loss, stats

    def _masked_cross_entropy(self, logits, target, length):
        return masked_cross_entropy(logits, target, length, padding_idx=self.padding_idx)


    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(batch_size, -1, _v.size(1))

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`mtdg.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return mtdg.utils.Statistics(loss.item(), num_non_padding, num_correct)



    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)


# https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def masked_cross_entropy(logits, target, length, per_example=False, padding_idx=1):
    """
    Args:
        logits (Variable, FloatTensor): [batch, max_len, num_classes]
            - unnormalized probability for each class
        target (Variable, LongTensor): [batch, max_len]
            - index of true class for each corresponding step
        length (Variable, LongTensor): [batch]
            - length of each data in a batch
    Returns:
        loss (Variable): []
            - An average loss value masked by the length
    """
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    logits_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]
    log_probs_flat = F.log_softmax(logits_flat, dim=1)

    # [batch_size * max_len, 1]
    target_flat = target.view(-1, 1)

    # Negative Log-likelihood: -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # [batch_size, max_len]
    losses = losses_flat.view(batch_size, max_len)

    # [batch_size, max_len]
    mask = _sequence_mask(sequence_length=length, max_len=max_len)

    # Apply masking on loss
    losses = losses * mask.float()

    # word-wise cross entropy
    # loss = losses.sum() / length.float().sum()

    target = target_flat.squeeze()
    pred = log_probs_flat.max(1)[1]
    non_padding = target.ne(padding_idx)
    num_correct = pred.eq(target) \
        .masked_select(non_padding) \
        .sum() \
        .item()
    num_non_padding = non_padding.sum().item()

    loss = losses.sum()
    loss.backward()

    return mtdg.utils.Statistics(loss.item(), num_non_padding, num_correct)

    # if per_example:
    #     # loss: [batch_size]
    #     return losses.sum(1)
    # else:
    #     loss = losses.sum()
    #     return loss, length.float().sum()



class TopicLossCompute(nn.Module):
    def __init__(self, tgt_vocab):
        super(TopicLossCompute, self).__init__()
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[PAD_TOKEN]
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = nn.BCELoss(weight, size_average=False)
        self.criterion2 = nn.CrossEntropyLoss(weight, size_average=False)

    def forward(self, scores, target, batch_size, train=True):
        # transform tokens to one-hot vectors
        # target = torch.Tensor(len(input_sentences), len(self.tgt_vocab)).zero_()
        # for idx, x in enumerate(input_sentences):
        #     target[idx][x] = 1
        # target = target.type_as(scores).contiguous()

        # batch_size = target.size(0)
        loss = self.criterion(scores, target)
        loss_data = loss.data.clone()
        if train:
            loss.div(batch_size).backward()
            # loss.backward()
        return loss_data