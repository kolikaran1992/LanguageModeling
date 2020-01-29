from keras.callbacks import Callback
import keras.backend as K
import numpy as np
from seqeval.metrics import f1_score, classification_report, precision_score, recall_score
from .__logger__ import LOGGER_NAME
import logging

logger = logging.getLogger(LOGGER_NAME+'_file')


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
        self._losses['val'].append(logs['val_loss'])
        self._losses['train'].append(logs['loss'])


class F1Metrics(Callback):

    def __init__(self, id2label, pad_value=0, validation_data=None, digits=4):
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
        self.validation_data = validation_data
        self.digits = digits
        self.is_fit = False

    def convert_idx_to_name(self, y, array_indexes):
        """Convert label index to name.
        Args:
            y (np.ndarray): label index 2d array.
            array_indexes (list): list of valid index arrays for each row.
        Returns:
            y: label name list.
        """
        y = [[self.id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    def predict(self, X, y):
        """Predict sequences.
        Args:
            X (np.ndarray): input data.
            y (np.ndarray): tags.
        Returns:
            y_true: true sequences.
            y_pred: predicted sequences.
        """
        y_pred = self.model.predict_on_batch(X)

        # reduce dimension.
        y_true = np.argmax(y, -1)
        y_pred = np.argmax(y_pred, -1)

        non_pad_indexes = [np.nonzero(y_true_row != self.pad_value)[0] for y_true_row in y_true]

        y_true = self.convert_idx_to_name(y_true, non_pad_indexes)
        y_pred = self.convert_idx_to_name(y_pred, non_pad_indexes)

        return y_true, y_pred

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
        if self.digits:
            print(classification_report(y_true, y_pred, digits=self.digits))
        return f1, precision, recall

    def on_epoch_end(self, epoch, logs={}):
        if self.is_fit:
            self.on_epoch_end_fit(epoch, logs)
        else:
            self.on_epoch_end_fit_generator(epoch, logs)

    def on_epoch_end_fit(self, epoch, logs={}):
        X = all_val_data[0]
        y = all_val_data[1]
        y_true, y_pred = self.predict(X, y)
        f1, precision, recall = self.score(y_true, y_pred)
        logs['val_f1'] = f1
        logs['val_precision'] = precision
        logs['val_recall'] = recall
        # y_true, y_pred = self.predict(train_tokens, train_labels)
        # f1, prec, rec = self.score(y_true, y_pred)
        # logs['f1'] = f1
        # logs['precision'] = prec
        # logs['recall'] = rec

    def on_epoch_end_fit_generator(self, epoch, logs={}):
        y_true = []
        y_pred = []
        for X, y in val_gen:
            y_true_batch, y_pred_batch = self.predict(X, y)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
        f1, precision, recall = self.score(y_true, y_pred)
        logs['val_f1'] = f1
        logs['val_precision'] = precision
        logs['val_recall'] = recall
