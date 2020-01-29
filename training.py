from .model import LanguageModel
from .data_prep import Processor
from time import gmtime, strftime
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from .__callbacks__ import *
from .__paths__ import path_to_lm_wts
from .__logger__ import LOGGER_NAME
import logging
import numpy as np

logger = logging.getLogger(LOGGER_NAME + '_console')

from keras.utils import Sequence
import math


def get_train_val_idxs(length, val_size, random_state):
    train_idxs, val_idxs, _, _ = train_test_split(
        [idx for idx in range(length)], [None] * length, test_size=val_size,
        random_state=random_state)
    logger.info('total train samples = {}'.format(len(train_idxs)))
    logger.info('total validation samples = {}'.format(len(val_idxs)))
    return train_idxs, val_idxs


class BatchGenerator(Sequence):
    def __init__(self, all_texts, text_processor=None, shuffle=True, batch_size=32):
        self.all_texts = all_texts
        self.batch_size = batch_size
        self._text_processor = text_processor
        self.shuffle = shuffle

    def __getitem__(self, idx):
        batch_ = self.all_texts[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_texts = [item[0] for item in batch_]
        batch_jtypes = [item[1] for item in batch_]
        toks, chars, jtypes, outs = self._text_processor.get_inputs_outputs(batch_texts, batch_jtypes)
        return [toks, chars, jtypes], outs

    def __len__(self):
        return math.ceil(len(self.all_texts) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.all_texts)


class Training(object):
    def __init__(self,
                 word2vec_path=None,
                 max_seq_len=None,
                 max_word_len=None,
                 all_jtypes=None,
                 char_emb_dim=16,
                 jtype_emb_size=8,
                 char_cnn_filters=256,
                 char_cnn_ker_size=8,
                 char_cnn_pool_size=8,
                 random_state=37,
                 name=None,
                 tensorboard_log_path=None,
                 save_peroid=5
                 ):
        self._save_period = save_peroid
        self._tensorboard_log_path = tensorboard_log_path
        self._name = name
        self._random_state = random_state
        self._processor = Processor(
            word2vec_path=word2vec_path,
            max_len=max_seq_len,
            max_char_len=max_word_len,
            all_jtypes=all_jtypes
        )

        logger.info('initialized text processor successfully')

        self._lm = LanguageModel(
            word_embedding_size=self._processor.get_vector_dim(),
            char_embedding_size=char_emb_dim,
            jtype_emb_size=jtype_emb_size,
            word_emb_weights=self._processor.get_vectors(),
            word_inp_mask_val=self._processor.get_vocab2int()['<pad>'],
            jtype_inp_mask_val=self._processor.get_jtype2int_dict()['<pad>'],
            jtype_vocab_size=len(self._processor.get_jtype2int_dict()),
            word_vocab_size=len(self._processor.get_vocab2int()),
            char_vocab_size=len(self._processor.get_char_vocab2int()),
            max_seq_len=max_seq_len,
            max_word_len=max_word_len,
            char_cnn_filters=char_cnn_filters,
            char_cnn_ker_size=char_cnn_ker_size,
            char_cnn_pool_size=char_cnn_pool_size
        )
        logger.info('initialized word2vec successfully')

    def compile_model(self, optimizer=None, loss=None, metrics=None):
        self._lm.get_model().compile(optimizer, loss=loss, metrics=metrics)

    def get_model_summary(self):
        self._lm.get_model().summary()

    def train(self, data, batch_size=256, initial_epoch=0, epochs=100):
        train_idxs, val_idxs = get_train_val_idxs(len(data), 0.2, self._random_state)
        train_gen = BatchGenerator(
            [data[idx] for idx in train_idxs],
            text_processor=self._processor, shuffle=True,
            batch_size=batch_size
        )

        val_gen = BatchGenerator(
            [data[idx] for idx in val_idxs],
            text_processor=self._processor, shuffle=True,
            batch_size=batch_size
        )
        path = path_to_lm_wts.joinpath(self._name, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
        saver = ModelCheckpoint(path, monitor='val_perplexity', verbose=0, save_best_only=False, save_weights_only=True,
                                mode='auto', period=self._save_period)
        tensorboard = self._tensorboard_log_path.joinpath(self._name,
                                                          '{}'.format(format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))))
        callbacks = [saver, tensorboard, Perplexity()]
        self.compile_model()
        self._processor.save(self._name)
        self._lm.get_model().fit_generator(train_gen, steps_per_epoch=len(train_gen),
                                           epochs=epochs, verbose=0, validation_data=val_gen,
                                           validation_steps=len(val_gen), initial_epoch=initial_epoch,
                                           callbacks=callbacks)
