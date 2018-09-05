import os
import torch
import torch.nn as nn

from collections import deque
from shutil import copyfile
from mtdg.utils.logging import logger
import mtdg.data

def build_model_saver(model, opt, fields, optim):
    model_saver = ModelSaver(opt.save_model, model, opt, fields, optim,
                             opt.save_checkpoint_epochs, opt.keep_checkpoint)
    return model_saver


class ModelSaverBase(object):
    """
        Base class for model saving operations
        Inherited classes must implement private methods:
            * `_save`
            * `_rm_checkpoint
    """

    def __init__(self, base_path, model, opt, fields, optim, save_checkpoint_epochs, keep_checkpoint=-1):
        self.base_path = base_path
        self.model = model
        self.opt = opt
        self.fields = fields
        self.optim = optim
        self.keep_checkpoint = keep_checkpoint
        self.save_checkpoint_epochs = save_checkpoint_epochs
        self.best_checkpoint = None
        self.minimum_valid_loss = float("inf")

        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def reset(self):
        self.best_checkpoint = None
        self.minimum_valid_loss = float("inf")
        if self.keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=self.keep_checkpoint)

    def maybe_save(self, epoch, valid_loss=None):
        """
        Main entry point for model saver
        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """
        if self.keep_checkpoint == 0:
            return

        if epoch % self.save_checkpoint_epochs != 0:
            return

        chkpt, chkpt_name = self._save(epoch)

        if valid_loss is not None and valid_loss < self.minimum_valid_loss:
            self.minimum_valid_loss = valid_loss
            if self.best_checkpoint is not None:
                self._rm_checkpoint(self.best_checkpoint)
            # self.best_checkpoint = chkpt_name + ".best_ppl_" + str(self.optim.learning_rate) + "." + str(valid_loss)
            self.best_checkpoint = "%s.best_lr_%5.4f_ppl_%5.2f" % (chkpt_name, self.optim.learning_rate, valid_loss)
            logger.info("Saving checkpoint %s with lowest ppl" % (self.best_checkpoint))
            copyfile(chkpt_name, self.best_checkpoint)

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, epoch):
        """ Save a resumable checkpoint.
        Args:
            epoch (int): epoch number
        Returns:
            checkpoint: the saved object
            checkpoint_name: name (or path) of the saved checkpoint
        """
        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """
        Remove a checkpoint
        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        raise NotImplementedError()

    def _cp_checkpoint(self, src_file_name, tgt_file_name):
        """
        copy a checkpoint as the best checkpoint
        Args:
            src_file_name(str): name that indentifies the checkpoint
                (it may be a filepath)
            tgt_file_name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """
        Simple model saver to filesystem
    """

    def __init__(self, base_path, model, opt, fields, optim, save_checkpoint_epochs, keep_checkpoint=0):
        super(ModelSaver, self).__init__(base_path, model, opt, fields, optim, save_checkpoint_epochs, keep_checkpoint)

    def _save(self, epoch):
        model_state_dict = self.model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'vocab': mtdg.data.save_fields_to_vocab(self.fields),
            'opt': self.opt,
            'epoch': epoch,
            'optim': self.optim
        }

        logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, epoch))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, epoch)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)

    def _cp_checkpoint(self, src_file_name, tgt_file_name):
        copyfile(src_file_name, tgt_file_name)