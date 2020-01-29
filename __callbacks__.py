from keras.callbacks import Callback
import keras.backend as K
import numpy as np
from seqeval.metrics import f1_score, classification_report, precision_score, recall_score
from seqeval.callbacks import F1Metrics
from .__logger__ import LOGGER_NAME
import logging
import os
import tensorflow as tf
from keras.callbacks import TensorBoard

logger_file = logging.getLogger(LOGGER_NAME + '_file')
logger_console = logging.getLogger(LOGGER_NAME + '_console')


class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self._losses = {'val': [], 'train': []}

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        try:  # new API
            lr = self.schedule(epoch, lr, self._losses)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        logger_console.info('{} : learning rate : {}'.format('LR', logs['lr']))
        self._losses['val'].append(logs['val_loss'])
        self._losses['train'].append(logs['loss'])


class Metrics(F1Metrics):

    def __init__(self, id2label, pad_value=0, val_data=None, val_gen=None, digits=4):
        """
        Args:
            id2label (dict): id to label mapping.
            (e.g. {1: 'B-LOC', 2: 'I-LOC'})
            pad_value (int): padding value.
            digits (int or None): number of digits in printed classification report
              (use None to print only F1 score without a report).
        """
        super(F1Metrics, self).__init__()
        self.id2label = id2label
        self.pad_value = pad_value
        self._val_data = val_data
        self._val_gen = val_gen
        self._digits = digits
        self._is_fit = False if val_gen else True if val_data else None

    def score(self, y_true, y_pred):
        """Calculate f1 score.
        Args:
            y_true (list): true sequences.
            y_pred (list): predicted sequences.
        Returns:
            score: f1 score.
        """
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        if self._digits:
            logger_file.info(classification_report(y_true, y_pred, digits=self._digits))
        return f1, precision, recall

    def on_epoch_end(self, epoch, logs={}):
        if self._is_fit:
            self.on_epoch_end_fit(epoch, logs)
        else:
            self.on_epoch_end_fit_generator(epoch, logs)

    def on_epoch_end_fit(self, epoch, logs={}):
        X = self._val_data[0]
        y = self._val_data[1]
        y_true, y_pred = self.predict(X, y)
        f1, precision, recall = self.score(y_true, y_pred)
        logs['val_f1'] = f1
        logs['val_precision'] = precision
        logs['val_recall'] = recall

    def on_epoch_end_fit_generator(self, epoch, logs={}):
        y_true = []
        y_pred = []
        for X, y in self._val_gen:
            y_true_batch, y_pred_batch = self.predict(X, y)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
        f1, precision, recall = self.score(y_true, y_pred)
        logs['val_f1'] = f1
        logs['val_precision'] = precision
        logs['val_recall'] = recall


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class Perplexity(Callback):
    def on_epoch_end(self, epoch, logs={}):
        super(Perplexity, self).on_epoch_end(epoch, logs=logs)
        logger_console.info(
            'epoch {} end || train_perplexity : {} || val_perplexity : {}'.format(epoch + 1, 2 ** logs['loss'],
                                                                                  2 ** logs['val_loss']))
        logs['perplexity'] = 2 ** logs['loss']
        logs['val_perplexity'] = 2 ** logs['val_loss']
