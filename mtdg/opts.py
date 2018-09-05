import argparse
import os

def model_opts(parser):
    """
        These options are passed to the construction of the model.
        Be careful with these as they will be used during translation.
        """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument('-src_word_vec_size', type=int, default=300, help='Word embedding size for src.')
    group.add_argument('-tgt_word_vec_size', type=int, default=300, help='Word embedding size for tgt.')
    group.add_argument('-word_vec_size', type=int, default=300, help='Word embedding size')

    group.add_argument('-share_decoder_embeddings', action='store_true', help="""Use a shared weight matrix for the input and output word  embeddings in the decoder.""")
    group.add_argument('-share_embeddings', action='store_true', help="""Share the word embeddings between encoder and decoder. Need to use shared dictionary for this option.""")
    group.add_argument('-position_encoding', action='store_true', help="""Use a sin to mark relative words positions. Necessary for non-RNN style models. """)

    # Encoder-Deocder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add_argument('-model', type=str, default='HRED', choices=['HRED', 'VHRED', 'HCRD', "TDCM", "TDACM"], help="""The gate type to use in the RNNs""")
    # General options.
    group.add_argument('-layers', type=int, default=-1, help='Number of layers in enc/dec.')
    group.add_argument('-rnn_size', type=int, default=500, help='Size of rnn hidden states')
    # Encoder
    group.add_argument('-enc_rnn_type', type=str, default='GRU', choices=['LSTM', 'GRU'], help="""The gate type to use in the RNNs""")
    group.add_argument('-enc_rnn_size', type=int, default=512, help='Size of rnn hidden states in the encoder')
    group.add_argument('-enc_layer', type=int, default=1, help='Number of layers in the encoder')
    group.add_argument('-bidirectional', action="store_true", help=""" . """)
    # Context_encoder
    group.add_argument('-context_rnn_type', type=str, default='GRU', choices=['LSTM', 'GRU'], help="""The gate type to use in the RNNs""")
    group.add_argument('-context_rnn_size', type=int, default=512, help='Size of rnn hidden states in the encoder')
    group.add_argument('-context_layer', type=int, default=1, help='Number of layers in the encoder')
    # Decoder
    group.add_argument('-dec_rnncell_type', type=str, default='GRU', choices=['LSTM', 'GRU'], help="""The gate type to use in the RNNs""")
    group.add_argument('-dec_rnn_size', type=int, default=512, help='Size of rnn hidden states in the encoder')
    group.add_argument('-dec_layer', type=int, default=1, help='Number of layers in the decoder')
    group.add_argument('-sample', action="store_true", help="""""")

    # Topic Encoder
    group.add_argument('-enc_cnn_type', type=str, default='rnn', choices=['base', 'gate', 'rnn'], help="""The gate type to use in the RNNs""")
    group.add_argument('-topic_rnn_type', type=str, default='GRU', choices=['LSTM', 'GRU'], help="""The gate type to use in the RNNs""")
    group.add_argument('-topic_rnn_size', type=int, default=512, help='Size of rnn hidden states in the encoder')
    group.add_argument('-topic_layer', type=int, default=1, help='Number of layers in the encoder')


    # group.add_argument('-topic', action='store_true', help="""""")
    group.add_argument('-topic_gate', action='store_true', help="""""")
    group.add_argument('-topic_num', type=int, default=100, help="""""")
    group.add_argument('-topic_size', type=int, default=100, help="""""")
    group.add_argument('-topic_key_size', type=int, default=300, help="""""")
    group.add_argument('-topic_value_size', type=int, default=300, help="""""")

    group.add_argument('-cnn_kernel_size', type=int, default=100, help="""Size of windows in the cnn, the kernel_size is (cnn_kernel_width, 1) in conv layer""")
    group.add_argument('-cnn_kernel_width', type=list, default=[2,3,4], help="""Size of windows in the cnn, the kernel_size is (cnn_kernel_width, 1) in conv layer""")
    # group.add_argument('-cnn_kernels', type=list, default=[2], help="""Size of windows in the cnn, the kernel_size is (cnn_kernel_width, 1) in conv layer""")

    parser.add_argument('-dropout', type=float, default=0.2)

    group.add_argument('-input_feed', type=int, default=1, help="""Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.""")
    group.add_argument('-bridge', action="store_true", help="""Have an additional layer between the last encoder state and the first decoder state""")
    group.add_argument('-residual',   action="store_true", help="Add residual connections between RNN layers.")
    group.add_argument('-context_gate', type=str, default=None, choices=['source', 'target', 'both'], help="""Type of context gate to use. Do not select for no context gate.""")


def preprocess_opts(parser):
    """" Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('-data', default='ubuntu', choices=["ubuntu", "dailydialog", "twitter", "opensub"])

    group.add_argument('-train_data', default="data/ubuntu_simp/raw_training_text.txt", help="Path to the training source data")
    group.add_argument('-valid_data', default="data/ubuntu_simp/raw_valid_text.txt", help="Path to the validation source data")
    group.add_argument('-test_data', default="data/ubuntu_simp/raw_test_text.txt", help="Path to the test source data")
    group.add_argument('-save_data', default="data/ubuntu_simp/convs")

    # Dictionary options, for text corpus
    group = parser.add_argument_group('Vocab')
    group.add_argument('-vocab', default="", help="""Path to an existing vocabulary. Format: one word per line.""")
    group.add_argument('-vocab_size', type=int, default=25000, help="Size of the source vocabulary")
    group.add_argument('-words_min_frequency', type=int, default=0)

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add_argument('-max_seq_length', type=int, default=30, help="Maximum sequence length.")
    group.add_argument('-min_seq_length', type=int, default=3, help="Truncate sequence length.")
    group.add_argument('-max_turn_length', type=int, default=12, help="Maximum sequence length.")
    group.add_argument('-min_turn_length', type=int, default=3, help="Maximum sequence length.")
    group.add_argument('-lower', action='store_true', help='lowercase data')
    group.add_argument('-max_turns', type=int, default=100, help="Maximum dialog (text context) length")

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('-shuffle', type=int, default=1, help="Shuffle data")
    group.add_argument('-seed', type=int, default=5381, help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=100, help="Report status every this many sentences")
    group.add_argument('-log_file', type=str, default="", help="Output logs to a file under this path.")

    group.add_argument('-n_workers', type=int, default=os.cpu_count())

def train_opts(parser):
    """ Training and saving options """
    group = parser.add_argument_group('General')
    group.add_argument('-data', required=True, help="""Path prefix to the ".train.pt" and ".valid.pt" file path from preprocess.py""")
    group.add_argument('-save_model', default='model', help="""Model filename (the model will be saved as <save_model>_N.pt where N is the number of epochs""")
    group.add_argument('-save_checkpoint_epochs', type=int, default=1, help="""Save a checkpoint every X epochs""")
    group.add_argument('-keep_checkpoint', type=int, default=1, help="""Keep X checkpoints (negative: keep all)""")

    # GPU
    group.add_argument('-gpuid', default=[], nargs='+', type=int, help="Use CUDA on the listed devices.")
    # group.add_argument('-gpu_rank', default=0, nargs='+', type=int, help="Rank the current gpu device.")
    # group.add_argument('-device_id', default=0, nargs='+', type=int, help="Rank the current gpu device.")
    # group.add_argument('-gpu_backend', default='nccl', nargs='+', type=str, help="Type of torch distributed backend")
    # group.add_argument('-gpu_verbose_level', default=0, type=int, help="Gives more info on each process per GPU.")
    group.add_argument('-seed', type=int, default=-1, help="""Random seed used for the experiments reproducibility.""")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add_argument('-param_init', type=float, default=0.1, help="""Parameters are initialized over uniform distribution with support (-param_init, param_init). Use 0 to not use initialization""")
    group.add_argument('-param_init_glorot', action='store_true', help="""Init parameters with xavier_uniform. Required for transfomer.""")
    group.add_argument('-train_from', type=str, help="""If training from a checkpoint then this is the path to the pretrained model's state_dict.""")

    # Pretrained word vectors
    group.add_argument('-pre_word_vecs_enc', help="""If a valid path is specified, then this will load pretrained word embeddings on the encoder side. See README for specific formatting instructions.""")
    group.add_argument('-pre_word_vecs_dec', help="""If a valid path is specified, then this will load pretrained word embeddings on the decoder side. See README for specific formatting instructions.""")
    # Fixed word vectors
    group.add_argument('-fix_word_vecs_enc', action='store_true', help="Fix word embeddings on the encoder side.")
    group.add_argument('-fix_word_vecs_dec', action='store_true', help="Fix word embeddings on the encoder side.")
    # Fixed utterance & conversation lenth
    group.add_argument('-fix_utterance_length', action='store_true', help="Fix word embeddings on the encoder side.")


    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument('-batch_size', type=int, default=64, help='Maximum batch size for training')
    group.add_argument('-batch_type', default='sents', choices=["sents", "tokens"], help="""Batch grouping for batch_size. Standard is sents. Tokens will do dynamic batching""")
    group.add_argument('-normalization', default='sents', choices=["sents", "tokens"], help='Normalization method of the gradient.')
    group.add_argument('-accum_count', type=int, default=1, help="""Accumulate gradient this many times. Approximately equivalent to updating batch_size * accum_count batches at once. Recommended for Transformer.""")
    group.add_argument('-valid_steps', type=int, default=100, help='Perfom validation every X steps')
    group.add_argument('-valid_batch_size', type=int, default=32, help='Maximum batch size for validation')
    group.add_argument('-max_generator_batches', type=int, default=16, help="""Maximum batches of words in a sequence to run the generator on in parallel. Higher is faster, but uses more memory.""")
    group.add_argument('-train_steps', type=int, default=3000, help='Number of training steps')
    group.add_argument('-start_epoch', type=int, default=0)
    group.add_argument('-epochs', type=int, default=100)

    group.add_argument('-optim', default='adam', choices=['sgd', 'adagrad', 'adadelta', 'adam','sparseadam'], help="""Optimization method.""")
    group.add_argument('-adagrad_accumulator_init', type=float, default=0, help="""Initializes the accumulator values in adagrad. Mirrors the initial_accumulator_value option in the tensorflow adagrad (use 0.1 for their default). """)
    group.add_argument('-max_grad_norm', type=float, default=5, help="""If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm""")
    group.add_argument('-truncated_decoder', type=int, default=0, help="""Truncated bptt.""")
    group.add_argument('-adam_beta1', type=float, default=0.9, help="""The beta1 parameter used by Adam. Almost without exception a value of 0.9 is used in the literature, seemingly giving good results, so we would discourage changing this value from the default without due consideration.""")
    group.add_argument('-adam_beta2', type=float, default=0.999, help="""The beta2 parameter used by Adam. Typically a value of 0.999 is recommended, as this is the value suggested by the original paper describing Adam, and is also the value adopted in other frameworks such as Tensorflow and Kerras, i.e. see: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer https://keras.io/optimizers/ . Whereas recently the paper "Attention is All You Need" suggested a value of 0.98 for beta2, this parameter may not work well for normal models / default baselines.""")
    group.add_argument('-label_smoothing', type=float, default=0.0, help="""Label smoothing value epsilon. Probabilities of all non-true labels will be smoothed by epsilon / (vocab_size - 1). Set to zero to turn off label smoothing. For more detailed information, see: https://arxiv.org/abs/1512.00567""")
    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument('-learning_rate', type=float, default=0.0001, help="""Starting learning rate. Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
    group.add_argument('-learning_rate_decay', type=float, default=0.5, help="""If update_learning_rate, decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) steps have gone past start_decay_steps""")
    group.add_argument('-start_decay_steps', type=int, default=None, help="""Start decaying every decay_steps after start_decay_steps""")
    group.add_argument('-decay_steps', type=int, default=10000, help="""Decay every decay_steps""")
    group.add_argument('-decay_method', type=str, default="", choices=['noam'], help="Use a custom decay rate.")
    group.add_argument('-warmup_steps', type=int, default=4000, help="""Number of warmup steps for custom decay.""")


def generate_opts(parser):
    """ Generate / inference options """
    group = parser.add_argument_group('Model')
    group.add_argument('-ckpt', type=str, default='ckpts/ubuntu_simp/TDACM_step_30.pt')

    group = parser.add_argument_group('Data')
    group.add_argument('-data', default='data/ubuntu_simp/convs.test.pt', help="""Path """)
    group.add_argument('-target', default='prdcs/ubuntu_simp/target.TDACM_step_30.pt.txt')
    group.add_argument('-output', default='prdcs/ubuntu_simp/output.TDACM_step_30.pt.txt')

    group = parser.add_argument_group('Efficiency')
    group.add_argument('-batch_size', type=int, default=20, help='Batch size')
    group.add_argument('-gpu', type=int, default=0, help="Device to run on")


def evaluate_opts(parser):
    """ Evaluation options """
    group = parser.add_argument_group('Evaluation')
    group.add_argument('-report_ppl', action='store_true', help="Perplexity")
    group.add_argument('-report_xent', action='store_true', help="xent Cross Entropy Loss")
    group.add_argument('-report_bleu', action='store_true', help="BLEU")
    group.add_argument('-report_rouge', action='store_true', help="ROUGE")
    group.add_argument('-report_accuracy', action='store_true', help="word accuracy")
    group.add_argument('-report_distinct', action='store_true', help="distinct n-gram (diversity)")
    group.add_argument('-report_embedding', action='store_true', help="Embedding Metric Average | Mean | Greedy")
    group.add_argument('-embeddings', type=str, default="data/GoogleNews-vectors-negative300.bin", help="embeddings bin file")


