def read_and_tokenize(dialog_path, min_turn=3):
    """
    Read conversation
    Args:
        dialog_path (str): path of dialog (tsv format)
    Return:
        dialogs: (list of list of str) [dialog_length, sentence_length]
        users: (list of str); [2]
    """
    with open(dialog_path, 'r', encoding='utf-8') as f:

        # Go through the dialog
        first_turn = True
        dialog = []
        users = []
        same_user_utterances = []  # list of sentences of current user
        dialog.append(same_user_utterances)

        for line in f:
            _time, speaker, _listener, sentence = line.split('\t')
            users.append(speaker)

            if first_turn:
                last_speaker = speaker
                first_turn = False

            # Speaker has changed
            if last_speaker != speaker:
                same_user_utterances = []
                dialog.append(same_user_utterances)

            same_user_utterances.append(sentence)
            last_speaker = speaker

        # All users in conversation (len: 2)
        users = list(OrderedDict.fromkeys(users))

        # 1. Concatenate consecutive sentences of single user
        # 2. Tokenize
        dialog = [tokenizer(" ".join(sentence)) for sentence in dialog]

        if len(dialog) < min_turn:
            print(f"Dialog {dialog_path} length ({len(dialog)}) < minimum required length {min_turn}")
            return []

    return dialog #, users