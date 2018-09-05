import torch
import torch.nn as nn

import mtdg.utils
from mtdg.loss import LanguageLossCompute, TopicLossCompute
from mtdg.utils.logging import logger
from mtdg.utils.misc import use_gpu

import time

def build_trainer(opt, model, fields, optim, device, model_saver=None, vis_logger=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    loss_function = LanguageLossCompute(fields["conversation"].vocab).to(device)
    shard_size = opt.max_generator_batches
    trainer = Trainer(model, fields, loss_function, optim, shard_size, model_saver=model_saver, vis_logger=vis_logger)
    return trainer


class Trainer(object):
    def __init__(self, model, fields, loss_function, optim, shard_size, model_saver=None, vis_logger=None):
        self.model = model
        self.fields = fields
        self.loss_function = loss_function
        self.optim = optim
        self.shard_size = shard_size
        self.model_saver = model_saver
        self.vis_logger = vis_logger

        self.model.train()


    def train(self, train_iter, valid_iter, train_epochs, valid_epochs, report_func=None,
              train_topic_iter=None, valid_topic_iter=None, topic_criterion=None, topic_optimizer=None):
        """
        The main training loops.
        """
        logger.info('Start training...')
        total_stats = mtdg.utils.Statistics()
        report_stats = mtdg.utils.Statistics()

        epoch = train_epochs[0]
        # for epoch in range(train_epochs):
        while epoch <= train_epochs[1]:

            # Pre-train with topics.
            # if train_topic_iter is not None:
                # self.train_topic(train_topic_iter, valid_topic_iter, epoch, topic_criterion, topic_optimizer)


            # training
            logger.info('Train conversations...')
            for i, batch in enumerate(train_iter):
                input_sentences, target_sentences = batch.conversation
                input_length, target_length = batch.length
                input_turns = batch.turn
                max_turn_length = input_turns.data.max()

                report_stats.n_src_words += input_length.sum().item()

                # self.optim.optimizer.zero_grad()
                self.model.zero_grad()

                scores = self.model(input_sentences, input_length, input_turns, target_sentences)

                batch_stats = self.loss_function.sharded_compute_loss(scores, target_sentences, self.shard_size)
                # batch_stats = self.loss_function._masked_cross_entropy(scores, target_sentences, target_length)

                self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                if i % 500 == -1 % 500:
                    report_stats.output(epoch, i + 1, len(train_iter), self.optim.learning_rate, total_stats.start_time)
                    # Visdom Logger
                    if self.vis_logger is not None:
                        self.vis_logger.line(X=torch.Tensor([i + epoch * len(train_iter)]),
                                              Y=torch.Tensor([report_stats.ppl()]),
                                              win="Training_perplexity", opts={"title": "Training Perplexity"},
                                              update=None if epoch == 0 and i == 9 else "append")
                        self.vis_logger.line(X=torch.Tensor([i + epoch * len(train_iter)]),
                                              Y=torch.Tensor([report_stats.xent()]),
                                              win="Training_xent", opts={"title": "Training Xent"},
                                              update=None if epoch == 0 and i == 9 else "append")
                        self.vis_logger.line(X=torch.Tensor([i + epoch * len(train_iter)]),
                                              Y=torch.Tensor([report_stats.accuracy()]),
                                              win="Training_accuracy", opts={"title": "Training Accuracy"},
                                              update=None if epoch == 0 and i == 9 else "append")
                    # self.visdom_logger([i + epoch * len(train_iter)], [report_stats.ppl()],
                    #                    win="")
                    report_stats = mtdg.utils.Statistics()

            print('Train xent: %g' % total_stats.xent())
            print('Train perplexity: %g' % total_stats.ppl())
            print('Train accuracy: %g' % total_stats.accuracy())
            # self.vis_logger.line(X=[epoch],
            #                      Y=[total_stats.ppl()],
            #                      win="Train_perplexity", opts={"title": "Train Perplexity"},
            #                      update=None if epoch == 0 and i == 0 else "append")
            # self.vis_logger.line(X=[epoch],
            #                      Y=[total_stats.xent()],
            #                      win="Train_xent", opts={"title": "Train Xent"},
            #                      update=None if epoch == 0 and i == 0 else "append")
            # self.vis_logger.line(X=[epoch],
            #                      Y=[total_stats.accuracy()],
            #                      win="Train_accuracy", opts={"title": "Train Accuracy"},
            #                      update=None if epoch == 0 and i == 0 else "append")

            # validation
            # if epoch % valid_epochs:
            valid_stats = self.valid(valid_iter)
            print('Validation xent: %g' % valid_stats.xent())
            print('Validation perplexity: %g' % valid_stats.ppl())
            print('Validation accuracy: %g' % valid_stats.accuracy())
            # self.vis_logger.line(X=[epoch],
            #                      Y=[valid_stats.ppl()],
            #                      win="Validation_perplexity", opts={"title": "Validation Perplexity"},
            #                      update=None if epoch == 0 and i == 0 else "append")
            # self.vis_logger.line(X=[epoch],
            #                      Y=[valid_stats.xent()],
            #                      win="Validation_xent", opts={"title": "Validation Xent"},
            #                      update=None if epoch == 0 and i == 0 else "append")
            # self.vis_logger.line(X=[epoch],
            #                      Y=[valid_stats.accuracy()],
            #                      win="Validation_accuracy", opts={"title": "Validation Accuracy"},
            #                      update=None if epoch == 0 and i == 0 else "append")

            # test
            # test_stats = self.test(test_iter)
            # print('Test xent: %g' % test_stats.xent())
            # print('Test perplexity: %g' % test_stats.ppl())
            # print('Test accuracy: %g' % test_stats.accuracy())
            # self.vis_logger.line(X=[epoch],
            #                      Y=[test_stats.ppl()],
            #                      win="Test_perplexity", opts={"title": "Test Perplexity"},
            #                      update=None if epoch == 0 and i == 0 else "append")
            # self.vis_logger.line(X=[epoch],
            #                      Y=[test_stats.xent()],
            #                      win="Test_xent", opts={"title": "Test Xent"},
            #                      update=None if epoch == 0 and i == 0 else "append")
            # self.vis_logger.line(X=[epoch],
            #                      Y=[test_stats.accuracy()],
            #                      win="Test_accuracy", opts={"title": "Test Accuracy"},
            #                      update=None if epoch == 0 and i == 0 else "append")

            if self.vis_logger is not None:
                self.vis_logger.line(X=torch.Tensor([[epoch, epoch]]),
                                     Y=torch.Tensor([[total_stats.ppl(), valid_stats.ppl()]]),
                                     win="ppl", opts={"title": "Perplexity"},
                                     update=None if epoch == 0 and i == 0 else "append")
                self.vis_logger.line(X=torch.Tensor([[epoch, epoch]]),
                                     Y=torch.Tensor([[total_stats.xent(), valid_stats.xent()]]),
                                     win="xent", opts={"title": "Xent"},
                                     update=None if epoch == 0 and i == 0 else "append")
                self.vis_logger.line(X=torch.Tensor([[epoch, epoch]]),
                                     Y=torch.Tensor([[total_stats.accuracy(), valid_stats.accuracy()]]),
                                     win="accuracy", opts={"title": "Accuracy"},
                                     update=None if epoch == 0 and i == 0 else "append")

            total_stats = mtdg.utils.Statistics()
            # drop checkpoints
            self._maybe_save(epoch, valid_stats.ppl())
            epoch += 1

        return total_stats


    def valid(self, valid_iter):
        """ Validate model.

                Returns:
                    :obj:`odlg.Statistics`: validation loss statistics
                """
        # Set model in validating mode.
        self.model.eval()
        stats = mtdg.utils.Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                input_sentences, target_sentences = batch.conversation
                input_length, target_length = batch.length
                turn = batch.turn
                stats.n_src_words += input_length.sum().item()

                # F-prop through the model.
                scores = self.model(input_sentences, input_length, turn, target_sentences)
                batch_stats = self.loss_function.monolithic_compute_loss(scores, target_sentences)

                # Update statistics.
                stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats


    def test(self, test_iter):
        """ Validate model.

                Returns:
                    :obj:`odlg.Statistics`: validation loss statistics
                """
        # Set model in validating mode.
        self.model.eval()
        stats = mtdg.utils.Statistics()

        with torch.no_grad():
            for batch in test_iter:
                input_sentences, target_sentences = batch.conversation
                input_length, target_length = batch.length
                turn = batch.turn
                stats.n_src_words += input_length.sum().item()

                # F-prop through the model.
                scores = self.model(input_sentences, input_length, turn, target_sentences)
                batch_stats = self.loss_function.monolithic_compute_loss(scores, target_sentences)

                # Update statistics.
                stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats


    def _maybe_save(self, epoch, valid_loss):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(epoch, valid_loss)



    def train_topic(self, train_iter, valid_iter, epoch, criterion=None, optimizer=None, test_iter=None):
        """
        Pretrain the topics.
        """
        logger.info('---------------------------------------------------------------------------------')
        logger.info('Start training topics...')

        # epoch = 0
        # for epoch in range(train_epochs):

        train_loss = 0
        # training
        for i, batch in enumerate(train_iter):
            # 1. Input
            input, length = batch.text
            target = batch.target
            input_sentences = input.t().contiguous()
            batch_size = len(batch)

            # 2. zero_grad
            self.model.zero_grad()

            # 3. Model
            topic_aware_representation, encoder_hidden = self.model.encoder(input_sentences)
            # scores = self.model.decoder.softmax(self.model.decoder.out(topic_aware_representation))
            scores = self.model.predictor(topic_aware_representation)

            # 4. Loss
            loss = criterion(scores, target, batch_size)
            train_loss += loss.item()

            # 5. optimize
            optimizer.step()

            # 6. Logging
            if i % 2000 == -1 % 2000:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * batch_size, len(train_iter.dataset),
                           100. * i / len(train_iter),
                    loss.item()))
        print('Train loss of topic: %g' % (train_loss / len(train_iter.dataset)))

        # validation
        valid_loss = None
        if valid_iter is not None:
            self.model.eval()
            valid_loss = 0
            with torch.no_grad():
                for idx, batch in enumerate(valid_iter):
                    input, length = batch.text
                    target = batch.target
                    input_sentences = input.t().contiguous()
                    batch_size = len(batch)

                    topic_aware_representation, encoder_hidden = self.model.encoder(input_sentences, length)
                    scores = self.model.predictor(topic_aware_representation)
                    # scores = self.model.decoder.softmax(self.model.decoder.out(topic_aware_representation))

                    loss = criterion(scores, target, batch_size, train=False)
                    valid_loss += loss.item()

            self.model.train()
            valid_loss = valid_loss / len(valid_iter.dataset)
            print('Validation loss of topic: %g' % (valid_loss))

        # drop checkpoints
        # self._maybe_save(epoch, valid_loss)

    # def _may_stop(self, valid_loss):

    def train_topic_v2(self, train_iter, valid_iter, train_epochs, valid_epochs, criterion=None, optimizer=None, test_iter=None):
        """
        Pretrain the topics.
        """

        logger.info('Start training topics...')

        epoch = 0
        for epoch in range(train_epochs):

            train_loss = 0
            # training
            for i, batch in enumerate(train_iter):

                # 1. Input
                input_sentences, target_sentences = batch.conversation
                input_length, target_length = batch.length
                input_turns = batch.turn
                max_turn_length = input_turns.data.max()

                # self.optim.optimizer.zero_grad()
                self.model.zero_grad()

                # scores = self.model(input_sentences, input_length, input_turns, target_sentences)

                # 3. Model
                topic_hidden, encoder_hidden = self.model.encoder(input_sentences)
                start = torch.cumsum(torch.cat((input_turns.data.new(1).zero_(), input_turns[:-1])), 0)

                scores = self.model.decoder.softmax(self.model.decoder.out(topic_aware_representation))

                # 4. Loss
                loss = criterion(scores, target, batch_size)
                train_loss += loss.item()

                # 5. optimize
                optimizer.step()

                # 6. Logging
                if i % 300 == -1 % 300:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i * batch_size, len(train_iter.dataset),
                               100. * i / len(train_iter),
                        loss.item()))
            print('Train loss of topic: %g' % (train_loss / len(train_iter.dataset)))

            # validation
            self.model.eval()
            valid_loss = 0
            with torch.no_grad():
                for idx, batch in enumerate(valid_iter):
                    input, length = batch.text
                    target = batch.target
                    input_sentences = input.t().contiguous()
                    batch_size = len(batch)

                    topic_aware_representation, encoder_hidden = self.model.encoder(input_sentences, length)
                    scores = self.model.predictor(encoder_hidden.squeeze(0))

                    loss = criterion(scores, target, batch_size, train=False)
                    valid_loss += loss.item()

            self.model.train()
            valid_loss = valid_loss / len(valid_iter.dataset)
            print('Validation loss of topic: %g' % (valid_loss))

            # drop checkpoints
            self._maybe_save(epoch, valid_loss)

    def visdom_logger(self, X, Y, win, opts, epoch=None, step=None):
        update = None if epoch == 0 or step == 0 else "append"
        self.vis_logger.line(X=X,
                             Y=Y,
                             win=win, opts=opts,
                             update="append")