from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D, MaxPool1D, \
    Flatten, concatenate
from .__commons__ import *
from .__layers__ import *
from .__logger__ import LOGGER_NAME
import logging

logger = logging.getLogger(LOGGER_NAME + '_console')


class LanguageModel(object):
    def __init__(self,
                 word_embedding_size=None,
                 char_embedding_size=None,
                 jtype_emb_size=None,
                 word_emb_weights=None,
                 word_inp_mask_val=None,
                 jtype_inp_mask_val=None,
                 jtype_vocab_size=None,
                 word_vocab_size=None,
                 char_vocab_size=None,
                 max_seq_len=None,
                 max_word_len=None,
                 char_cnn_filters=None,
                 char_cnn_ker_size=None,
                 char_cnn_pool_size=None,
                 dropout=0.4,
                 rnn_dropout=0.4
                 ):
        self._model = None
        self._max_seq_len = max_seq_len
        self._max_word_len = max_word_len
        self._word_emb_size = word_embedding_size
        self._word_emb_wts = word_emb_weights
        self._char_emb_size = char_embedding_size
        self._jtype_emb_size = jtype_emb_size
        self._word_inp_mask = word_inp_mask_val
        self._jtype_inp_mask = jtype_inp_mask_val
        self._jtype_vocab_size = jtype_vocab_size
        self._word_vocab_size = word_vocab_size
        self._char_vocab_size = char_vocab_size
        self._char_cnn_filters = char_cnn_filters
        self._char_cnn_ker_size = char_cnn_ker_size
        self._char_cnn_pool_size = char_cnn_pool_size
        self._dropout = dropout
        self._rnn_dropout = rnn_dropout

    def _get_word_embedding_layer(self, word_in):
        masked_word_in = Mask(mask_value=self._word_inp_mask, name='masked_word_inputs')(word_in)
        if self._word_emb_wts:
            emb_word = Embedding(input_dim=self._word_vocab_size,
                                 output_dim=self._word_emb_size,
                                 input_length=self._max_seq_len,
                                 weights=[self._word_emb_wts],
                                 name='word_embeddings', trainable=False)(masked_word_in)
        else:
            emb_word = Embedding(input_dim=self._word_vocab_size,
                                 output_dim=self._word_emb_size,
                                 input_length=self._max_seq_len,
                                 name='word_embeddings', trainable=True)(masked_word_in)
        return emb_word

    def _get_char_embedding_layer(self, char_in):
        emb_char = TimeDistributed(Embedding(input_dim=self._char_vocab_size, output_dim=self._char_emb_size,
                                             input_length=self._max_word_len,
                                             name='char_embeddings'),
                                   name='time_distributed_char_emb',
                                   trainable=True)(char_in)
        emb_char = TimeDistributed(Conv1D(filters=self._char_cnn_filters,
                                          kernel_size=self._char_cnn_ker_size, activation='relu', name='char_cnn',
                                          trainable=True),
                                   name='time_distributed_char_cnn',
                                   trainable=True)(emb_char)
        emb_char = TimeDistributed(Dropout(self._dropout))(emb_char)
        emb_char = TimeDistributed(MaxPool1D(pool_size=self._char_cnn_pool_size), name='char_cnn_pooling')(emb_char)
        emb_char = TimeDistributed(Flatten(), name='char_cnn_flatten')(emb_char)

        return emb_char

    def _get_jtype_embedding_layer(self, jtype_in):
        masked_jtype_in = Mask(mask_value=self._jtype_inp_mask, name='masked_jtype_inputs')(jtype_in)
        emb_jtype = Embedding(input_dim=self._jtype_vocab_size,
                              output_dim=self._jtype_emb_size,
                              input_length=self._max_seq_len,
                              name='jtype_embeddings', trainable=True)(masked_jtype_in)
        return emb_jtype

    def _get_embedding_layer(self, word_in, char_in, jtype_in):
        word_emb = self._get_word_embedding_layer(word_in)
        char_emb = self._get_char_embedding_layer(char_in)
        jtype_emb = self._get_jtype_embedding_layer(jtype_in)

        merged_embedding = concatenate([word_emb, char_emb, jtype_emb], name='merge_char_word_jtype_embeddings')
        final_emb = PosEmb(self._max_seq_len, merged_embedding.shape[-1], name='final_embedding')(merged_embedding)

        return final_emb

    def _get_elmo_style_lstm_out(self, inp, name='', rev=False):
        _lstm1 = LSTM(units=inp.size[-1],
                      return_sequences=True,
                      recurrent_dropout=self._rnn_dropout, go_backwards=rev,
                      name='{}_1'.format(name), trainable=True)(inp)
        merged_inp = concatenate([inp, _lstm1], name='{}_merge_lstm1_embed'.format(name))
        _lstm2 = LSTM(units=inp.size[-1],
                      return_sequences=True,
                      recurrent_dropout=self._rnn_dropout,
                      name='{}_2'.format(name), trainable=True)(merged_inp)
        return _lstm2

    def model(self):
        if self._model:
            return self._model

        word_in = Input(shape=(self._max_seq_len,), name='token_inputs', dtype=dtype_int)
        char_in = Input(shape=(self._max_seq_len, self._max_word_len,), name='char_inputs', dtype=dtype_int)
        jtype_in = Input(shape=(self._max_seq_len,), name='jtype_inputs', dtype=dtype_int)

        embedding_layer = self._get_embedding_layer(word_in, char_in, jtype_in)

        l2r_lstm = self._get_elmo_style_lstm_out(embedding_layer, name='l2r_lstm', rev=False)
        r2l_lstm = self._get_elmo_style_lstm_out(embedding_layer, name='r2l_lstm', rev=True)
        merged = concatenate([l2r_lstm, r2l_lstm], name='merge_lstm1_lstm2')

        model = TimeDistributed(Dense(self._word_vocab_size, name='token_prediction', activation='softmax'))(
            merged)
        model = Model(inputs=[word_in, char_in, jtype_in], outputs=[model], name='LanguageModel')
        self._model = model
        return self._model
