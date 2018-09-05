import torch
import torch.nn as nn
import torch.nn.functional as F


class HRED(nn.Module):
    def __init__(self, encoder, context_encoder, context2decoder, decoder):
        super(HRED, self).__init__()
        self.encoder = encoder
        self.context_encoder = context_encoder
        self.decoder = decoder
        self.context2decoder = context2decoder

    def forward(self, input_sentences, input_lengths, input_turns, target_sentences, decode=False):
        """
        Args:
            input_sentences: (LongTensor) [num_sentences, max_utterance_length]
            target_sentences: (LongTensor) [num_sentences, max_utterance_length]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        num_sentences = input_sentences.size(0)
        max_turn_length = input_turns.data.max()


        # 1.encoder_outputs: [num_sentences, max_utterance_length, hidden_size * direction]
        #   encoder_hidden: [num_layers * direction, num_sentences, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(input_sentences, input_lengths)

        # encoder_hidden: [num_sentences, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(num_sentences, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((input_turns.data.new(1).zero_(), input_turns[:-1])), 0)

        # encoder_hidden: [batch_size, max_turn_length, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack(
            [torch.cat([encoder_hidden.narrow(0, s, l),
            encoder_hidden.data.new(max_turn_length.item()-l, encoder_hidden.size(1)).zero_()])
            for s, l in zip(start.data.tolist(), input_turns.data.tolist())], 0)


        # 2.context_outputs: [batch_size, context_size]
        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden, input_turns)

        # flatten outputs
        # context_outputs: [num_sentences, context_size]
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_turns.data)])


        # 3.decoder
        # project context_outputs to decoder init state
        decoder_init = self.context2decoder(context_outputs)

        # [num_layers, batch_size, hidden_size]
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        if not decode:

            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode)
            return decoder_outputs

        else:
            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode)
            return decoder_outputs.unsqueeze(1)
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)

            # Get top prediction only
            # [batch_size, max_unroll]
            # prediction = prediction[:, 0]

            # [batch_size, beam_size, max_unroll]
            return prediction



class TDCM(nn.Module):
    def __init__(self, encoder, context_encoder, context2decoder, decoder):
        super(TDCM, self).__init__()
        self.encoder = encoder
        self.context_encoder = context_encoder
        self.context2decoder = context2decoder
        self.decoder = decoder

    def forward(self, input_sentences, input_lengths, input_turns, target_sentences, decode=False):
        """
        Args:
            input_sentences: (LongTensor) [num_sentences, max_utterance_length]
            target_sentences: (LongTensor) [num_sentences, max_utterance_length]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        num_sentences = input_sentences.size(0)
        max_turn_length = input_turns.data.max()


        # 1.encoder_outputs: [num_sentences, topic_value_size]
        #   encoder_hidden: [num_layers * direction, num_sentences, hidden_size]
        topic_aware_representation, encoder_hidden = self.encoder(input_sentences, input_lengths)

        # encoder_hidden: [num_sentences, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(num_sentences, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((input_turns.data.new(1).zero_(), input_turns[:-1])), 0)

        # encoder_hidden: [batch_size, max_turn_length, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack(
            [torch.cat([encoder_hidden.narrow(0, s, l),
            encoder_hidden.data.new(max_turn_length.item()-l, encoder_hidden.size(1)).zero_()])
            for s, l in zip(start.data.tolist(), input_turns.data.tolist())], 0)


        # 2.context_outputs: [batch_size, context_size]
        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden, input_turns)

        # flatten outputs
        # context_outputs: [num_sentences, context_size]
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_turns.data)])


        # 3.decoder
        # project context_outputs to decoder init state
        decoder_init = self.context2decoder(context_outputs)

        # [num_layers, batch_size, hidden_size]
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        if not decode:

            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode,
                                           topic_rep=topic_aware_representation)
            return decoder_outputs

        else:
            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode,
                                           topic_rep=topic_aware_representation)
            return decoder_outputs.unsqueeze(1)
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)

            # Get top prediction only
            # [batch_size, max_unroll]
            # prediction = prediction[:, 0]

            # [batch_size, beam_size, max_unroll]
            return prediction


class TDACM(nn.Module):
    def __init__(self, encoder, context_encoder, topic_encoder, context2decoder, topic2decoder, decoder, self_attention=None):
        super(TDACM, self).__init__()
        self.encoder = encoder
        self.context_encoder = context_encoder
        self.topic_encoder = topic_encoder
        self.context2decoder = context2decoder
        self.topic2decoder = topic2decoder
        self.decoder = decoder
        self.self_attention = self_attention

    def forward(self, input_sentences, input_lengths, input_turns, target_sentences, decode=False):
        """
        Args:
            input_sentences: (LongTensor) [num_sentences, max_utterance_length]
            target_sentences: (LongTensor) [num_sentences, max_utterance_length]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        num_sentences = input_sentences.size(0)
        max_turn_length = input_turns.data.max()


        # 1.encoder_outputs: [num_sentences, topic_value_size]
        #   encoder_hidden: [num_layers * direction, num_sentences, hidden_size]
        topic_hidden, encoder_hidden = self.encoder(input_sentences, input_lengths)

        # encoder_hidden: [num_sentences, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(num_sentences, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((input_turns.data.new(1).zero_(), input_turns[:-1])), 0)

        # encoder_hidden: [batch_size, max_turn_length, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack(
            [torch.cat([encoder_hidden.narrow(0, s, l),
            encoder_hidden.data.new(max_turn_length.item()-l, encoder_hidden.size(1)).zero_()])
            for s, l in zip(start.data.tolist(), input_turns.data.tolist())], 0)

        topic_hidden = torch.stack(
            [torch.cat([topic_hidden.narrow(0, s, l),
            topic_hidden.data.new(max_turn_length.item()-l, topic_hidden.size(1)).zero_()])
            for s, l in zip(start.data.tolist(), input_turns.data.tolist())], 0)

        # 2.context_outputs: [batch_size, context_size]
        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden, input_turns)
        topic_outputs, topic_last_hidden = self.topic_encoder(topic_hidden, input_turns)

        # flatten outputs
        # context_outputs: [num_sentences, context_size]
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_turns.data)])
        topic_outputs = torch.cat([topic_outputs[i, :l, :]
                                     for i, l in enumerate(input_turns.data)])

        # 3.decoder
        # project context_outputs to decoder init state
        decoder_init = self.context2decoder(context_outputs)
        topic_states = self.topic2decoder(topic_outputs)

        # [num_layers, batch_size, hidden_size]
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        if not decode:

            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode,
                                           topic_rep=topic_states)
            return decoder_outputs

        else:
            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode,
                                           topic_rep=topic_states)
            return decoder_outputs.unsqueeze(1)
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)

            # Get top prediction only
            # [batch_size, max_unroll]
            # prediction = prediction[:, 0]

            # [batch_size, beam_size, max_unroll]
            return prediction