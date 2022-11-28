import torch
import torchtext

from collections import defaultdict
from collections import Counter
from collections import OrderedDict
from itertools import chain, count
import six
import codecs
from pathlib import Path
from urllib.request import urlretrieve
import tarfile

import mtdg
from mtdg.utils.logging import logger

PAD_TOKEN = '<blank>'
UNK = 0
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('data/')
ubuntu_dir = datasets_dir.joinpath('ubuntu/')

ubuntu_meta_dir = ubuntu_dir.joinpath('meta/')
dialogs_dir = ubuntu_dir.joinpath('dialogs/')

class Dataset(torchtext.data.Dataset):
    def __init__(self, examples, fields, **kwargs):

        keys = fields.keys()
        out_fields = [(k, fields[k]) for k in keys]
        example_values = [[ex[k] for k in keys] for ex in examples]
        out_examples = [torchtext.data.Example.fromlist(ex_values, out_fields)
                        for ex_values in example_values]
        super(Dataset, self).__init__(out_examples, out_fields, **kwargs)

    def sort_key(self, ex):
        """ Sort using length of conversation turns. """
        return -ex.turn

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.
    Returns:
        a single dictionary that has the union of these keys.
    """
    return dict(chain(*[d.items() for d in args]))

def _peek(seq):
    """
    Args:
        seq: an iterator.

    Returns:
        the first thing returned by calling next() on the iterator
        and an iterator created by re-chaining that value to the beginning
        of the iterator.
    """
    first = next(seq)
    return first, chain([first], seq)

class Field(torchtext.data.Field):
    def __init__(self, remain_list=False, fix_conversation_turn=None,  **kwargs):
        super(Field, self).__init__(**kwargs)

        self.remain_list = remain_list
        self.fix_conversation_turn = fix_conversation_turn

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if self.remain_list:
            return [len_list[-self.fix_conversation_turn:] for len_list in minibatch]
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(max(len(x) for x in conversation) for conversation in minibatch)
        else:
            max_len = self.fix_length - 1
        padded = []
        for x in minibatch:
            _p = []
            x = x if self.fix_conversation_turn is None else x[-self.fix_conversation_turn:]
            for u in x:
                _p.append(
                    # ([] if self.init_token is None else [self.init_token]) +
                    list(u[-max_len:] if self.truncate_first else u[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(u)))
            padded.append(_p)
        return padded

    def numericalize(self, arr, device=None, train=True):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr: List of tokenized and padded examples, or tuple of a padded
                list and a list of lengths if self.include_lengths is True.
            device: Device to create the Variable's Tensor on. Use -1 for
                CPU and None for the currently active GPU device. Default:
                None.
            train: Whether the batch is for a training set. If False, the
                Variable will be created with volatile=True. Default: True.
        """
        if self.sequential:
            conversations = [[[self.vocab.stoi[token] if token in self.vocab.stoi else 0 for token in x] for x in _arr] for _arr in arr]

            input_sentences = [sent for conv in conversations for sent in conv[:-1]]
            target_sentences = [sent for conv in conversations for sent in conv[1:]]
            input_sentences = torch.tensor(input_sentences, dtype=self.dtype, device=device)
            target_sentences = torch.tensor(target_sentences, dtype=self.dtype, device=device)

            # if not self.batch_first:
            #     input_sentences.t_().contiguous()
            #     target_sentences.t_().contiguous()

            return  input_sentences, target_sentences
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))

            if self.remain_list:
                input_length = [min(l+1, self.fix_length) for len_list in arr for l in len_list[:-1]]
                target_length = [min(l+1, self.fix_length) for len_list in arr for l in len_list[1:]]
                input_length = torch.tensor(input_length, dtype=self.dtype, device=device)
                target_length = torch.tensor(target_length, dtype=self.dtype, device=device)
                return input_length, target_length

            input_turn = [min(l-1, self.fix_conversation_turn-1) for l in arr]
            var = torch.tensor(input_turn, dtype=self.dtype, device=device)
            return var



class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Whether the batch is from a training set.

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None, train=True, is_hierarchical=True):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.train = train
            for (name, field) in dataset.fields.items():
                if field is not None:
                    setattr(self, name, field.numericalize(
                        # field.pad([x.__dict__[name] for x in data]),
                        field.pad([x.__dict__[name] for x in data], is_hierarchical, name),
                        device=device, train=train))

    @classmethod
    def fromvars(cls, dataset, batch_size, train=True, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch

    def __len__(self):
        return self.batch_size


def get_fields():
    fields = {}
    fields["conversation"] = Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, fix_length=30, fix_conversation_turn=10)
    fields["turn"] = Field(use_vocab=False, sequential=False, dtype=torch.long, fix_conversation_turn=10)
    fields["length"] = Field(remain_list=True, use_vocab=False, sequential=False, dtype=torch.long, fix_length=30, fix_conversation_turn=10)
    fields["indices"] = Field(use_vocab=False, sequential=False, dtype=torch.long, fix_conversation_turn=10)
    return  fields

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
    vocab = dict(vocab)
    fields = get_fields()
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields

def _load_fields(dataset, opt, checkpoint):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = load_fields_from_vocab(checkpoint['vocab'])
    else:
        fields = load_fields_from_vocab(torch.load(opt.data + '.vocab.pt'))
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    logger.info(' * vocabulary size == %d' % len(fields['conversation'].vocab))
    return fields


def read_dailydialog_file(path, max_turns=0, truncate=0):
    if path is None:
        return None

    with codecs.open(path, "r", "utf-8") as corpus_file:
        conversations = []
        for i, line in enumerate(corpus_file):
            utterances = line.strip().split("__eou__")
            if len(utterances[-1]) == 0 or utterances[-1] == "":
                utterances = utterances[:-1]
            if max_turns:
                utterances = utterances[-max_turns:]
            turn = len(utterances)
            if turn > 1:
                example, length = [], []
                for utterance in utterances:
                    words = utterance.split()
                    if truncate:
                        words = words[:truncate]
                    example.append(words)
                    length.append(len(words))
                # the (turn & length) of the last utterance are removed.
                conv_dict = {"conversation": example, "turn": turn, "length": length, "indices":i}
                conversations.append(conv_dict)
        return conversations

def prepare_ubuntu_data(datasets_dir, ubuntu_dir, ubuntu_meta_dir, dialogs_dir):
    """Download and unpack dialogs"""

    tar_filename = 'ubuntu_dialogs.tgz'
    url = 'http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/ubuntu_dialogs.tgz'
    tarfile_path = ubuntu_dir.joinpath(tar_filename)
    metadata_url = 'https://raw.githubusercontent.com/rkadlec/ubuntu-ranking-dataset-creator/master/src/meta/'

    if not datasets_dir.exists():
        datasets_dir.mkdir()
    if not ubuntu_dir.exists():
        ubuntu_dir.mkdir()
    if not ubuntu_meta_dir.exists():
        ubuntu_meta_dir.mkdir()

    # Prepare Dialog data
    if not dialogs_dir.joinpath("10/1.tst").exists():
        # Download Dialog tarfile
        if not tarfile_path.exists():
            print(f"Downloading {url} to {tarfile_path}")
            urlretrieve(url, tarfile_path)
            print(f"Successfully downloaded {tarfile_path}")

        # Unpack tarfile
        if not dialogs_dir.exists():
            print("Unpacking dialogs ... (This can take 5~10 mins.)")
            with tarfile.open(tarfile_path) as tar:
                
                import os
                
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=ubuntu_dir)
            print("Archive unpacked.")

    # Download metadata
    if not ubuntu_meta_dir.joinpath('trainfiles.csv').exists():
        print('Downloading metadata ... (This can take 5~10 mins.)')
        for filename in ['trainfiles.csv', 'valfiles.csv', 'testfiles.csv']:
            csv_path = ubuntu_meta_dir.joinpath(filename)
            print(f"Downloading {metadata_url+filename} to {csv_path}")
            urlretrieve(metadata_url + filename, csv_path)
            print(f"Successfully downloaded {csv_path}")

    print('Ubuntu Data prepared!')

def get_dialog_path_list(dataset='train'):
    if dataset == 'train':
        filename = 'trainfiles.csv'
    elif dataset == 'test':
        filename = 'testfiles.csv'
    elif dataset == 'valid':
        filename = 'valfiles.csv'
    with open(ubuntu_meta_dir.joinpath(filename)) as f:
        dialog_path_list = []
        for line in f:
            file, dir = line.strip().split(",")
            path = dialogs_dir.joinpath(dir, file)
            dialog_path_list.append(path)

    return dialog_path_list

def read_ubuntu_file(path, max_turns=0, truncate=0):

    # Download and unpack dialogs if necessary.
    prepare_ubuntu_data()

    # List of dialogs (tsv)
    dialog_path_list = get_dialog_path_list(path)

    if path is None:
        return None

    with codecs.open(path, "r", "utf-8") as corpus_file:
        conversations = []
        for i, line in enumerate(corpus_file):
            utterances = line.strip().split("__eou__")
            if len(utterances[-1]) == 0 or utterances[-1] == "":
                utterances = utterances[:-1]
            if max_turns:
                utterances = utterances[-max_turns:]
            turn = len(utterances)
            if turn > 1:
                example, length = [], []
                for utterance in utterances:
                    words = utterance.split()
                    if truncate:
                        words = words[:truncate]
                    example.append(words)
                    length.append(len(words))
                # the (turn & length) of the last utterance are removed.
                conv_dict = {"conversation": example, "turn": turn, "length": length, "indices":i}
                conversations.append(conv_dict)
        return conversations



class OrderedIterator(torchtext.data.Iterator):
    """ Ordered Iterator Class """
    def create_batches(self):
        """ Create batches """
        if self.train:
            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                    # for b in list(p_batch):
                        yield b
            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def build_iterator(corpus_type, fields, opt, is_train=True, is_topic=False):
    if is_topic:
        pt_file = opt.data + '.' + corpus_type + '.topic'
        dataset = torch.load(pt_file)
        _fields = mtdg.topic_dataset.get_fields()
        _fields["text"].vocab = _fields["target"].vocab = fields["conversation"].vocab
        dataset.fields = _fields
    else:
        pt_file = opt.data + '.' + corpus_type + '.pt'
        dataset = torch.load(pt_file)
        dataset.fields = fields

    logger.info('Loading %s dataset from %s, number of examples: %d' %
                (corpus_type, pt_file, len(dataset)))
    device = torch.device("cuda" if opt.gpuid else "cpu")
    # return torchtext.data.Iterator(dataset=dataset, batch_size=opt.batch_size, device=device,
    #                                train=is_train, sort=not is_train)
    return OrderedIterator(dataset=dataset, batch_size=opt.batch_size, device=device,
                           train=is_train, sort=False, repeat=False)