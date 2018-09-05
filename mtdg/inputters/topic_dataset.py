# coding: utf8
import torch
import torchtext.data
import torchtext.vocab

import re
import math
import codecs
from collections import Counter, defaultdict
from itertools import chain

class TopicDataset(torchtext.data.Dataset):
    def __init__(self, path, fields,
                 encoding='utf-8', **kwargs):
        fields = [(key, fields[key]) for key in fields]
        raw_examples = self._process_file(path)
        examples = []
        for line in raw_examples:
            line = line.strip()
            if " +++$+++ " in line:
                sentence, target = line.split(" +++$+++ ")
                examples.append(
                    torchtext.data.Example.fromlist(
                        [sentence, target], fields))

        super(TopicDataset, self).__init__(
            examples, fields, **kwargs)

    def _process_file(self, path):
        raw_conversations = []
        sentence_for_counter = []
        counter = Counter()
        if ".pt" in path:
            raw_conversations = [conv.conversation for conv in torch.load(path).examples]
            for conv in raw_conversations:
                counter.update([word for sent in conv for word in sent])
        else:
            lines = open(path, 'r').readlines()
            for line in lines:
                conversation = []
                sentences = line.rstrip().split("__eot__")
                for sentence in sentences:
                    words = sentence.rstrip().split()
                    conversation.append(words)
                    sentence_for_counter.append(words)
                raw_conversations.append(conversation)

            for words in sentence_for_counter:
                counter.update(words)

        stopwords = set([item.strip().lower() for item in codecs.open("tools/stopwords.txt", encoding='utf-8').readlines()])
        freqwords = set([item[0] for item in sorted(counter.most_common(int(float(len(counter) * 0.001))))])
        alpha_check = re.compile("[a-zA-Z]")
        symbols = set([w for w in counter if ((alpha_check.search(w) == None) or w.startswith("'"))])
        ignore = stopwords | freqwords | symbols | set(["n't"])

        examples = []
        for conversation in raw_conversations:
            for idx in range(len(conversation)-1):
                target_words = [w for w in conversation[idx+1] if w not in ignore]
                if len(target_words) > 2 and len(conversation[idx]) > 4:
                    examples.append(" ".join(conversation[idx]) + " +++$+++ " + " ".join(target_words) + "\n")

                # if len(target_words) > 2 and len(words) > 4:
                # for i in range(math.ceil(len(target_words) / 3)):
                #     seq = target_words[i * 3: i * 3 + 3]
                #     examples.append(" ".join(words) + " +++$+++ " + " ".join(seq) + "\n")
        return examples

    def sort_key(self, ex):
        return len(ex.text)


    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


class TopicField(torchtext.data.Field):
    def __init__(self, multi_label=False, **kwargs):
        self.multi_label = multi_label
        super(TopicField, self).__init__(**kwargs)

    def process(self, batch, device=None):
        if not self.multi_label:
            padded = self.pad(batch)
            tensor = self.numericalize(padded, device=device)
        else:
            tensor = torch.Tensor(len(batch), len(self.vocab)).zero_()
            arr = [[self.vocab.stoi[x] for x in ex] for ex in batch]
            for idx, x in enumerate(arr):
                tensor[idx][torch.LongTensor(x)] = 1
            tensor = tensor.cuda(device)
            if device == -1:
                tensor = tensor.contiguous()
            else:
                tensor = tensor.cuda(device)
        return tensor


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab

def load_fields_from_vocab(vocab):
    """
    Load Field objects from `vocab.pt` file.
    """
    fields = get_fields()
    vocab = dict(vocab)
    # fields = get_fields()
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields


def get_fields():
    fields = {}
    fields["text"] = TopicField(lower=True, include_lengths=True)
    fields["target"] = TopicField(lower=True, multi_label=True)
    return fields