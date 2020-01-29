from gensim.models import KeyedVectors
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from .__commons__ import *
from .__logger__ import LOGGER_NAME
import logging
from .__paths__ import path_to_processor
import pickle

logger = logging.getLogger(LOGGER_NAME + '_console')

lower = False


class Processor(object):
    def __init__(self,
                 word2vec_path=None,
                 max_len=None,
                 max_char_len=None,
                 all_jtypes=(),
                 tokenizer=None
                 ):
        if tokenizer:
            self._tokenizer = tokenizer
        else:
            reg = r'\w+|[^\w\s]'
            self._tokenizer = RegexpTokenizer(reg)
            logger.info('{}: using default regex ({}) tokenizer'.format(self.__class__.__name__, reg))

        self._max_len = max_len
        self._max_char_len = max_char_len
        if word2vec_path:
            self._w2v = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
            self._w2v.add(pad_tok, np.zeros((1, self._w2v.vector_size)))
            self._w2v.add(end_tok, np.random.random((1, self._w2v.vector_size)))
            self._w2v.add(unk_tok, np.random.random((1, self._w2v.vector_size)))
            self._vocab2int = {tok: i for i, tok in
                               enumerate(list(self._w2v.vocab.keys()))}
            self._int2vocab = {i: tok for i, tok in
                               enumerate(list(self._w2v.vocab.keys()))}
            self._char_vocab2int = {ch: idx for idx, ch in enumerate(
                [pad_tok] + list(set([ch for tok in list(self._vocab2int.keys())[1:-1] for ch in tok])) + [unk_tok])}
            self._char_int2vocab = {idx: ch for idx, ch in enumerate(
                [pad_tok] + list(set([ch for tok in list(self._vocab2int.keys())[1:-1] for ch in tok])) + [unk_tok])}
        else:
            logger.info('no word vector path provided, vectors and vocabs will not be initialized')

        if all_jtypes:
            self._jtype2int_dict = {jtype: idx for idx, jtype in enumerate([pad_tok] + all_jtypes + [unk_tok])}
            self._int2jtype_dict = {idx: jtype for idx, jtype in enumerate([pad_tok] + all_jtypes + [unk_tok])}
        else:
            logger.info('jtypes will not be initialized')

    def extract_tokens(self, text):
        return [tok for tok in self._tokenizer.tokenize(text.lower() if lower else text)]

    def get_vocab2int(self):
        return self._vocab2int

    def get_jtype2int_dict(self):
        return self._jtype2int_dict

    def get_char_vocab2int(self):
        return self._char_vocab2int

    def get_vector_dim(self):
        return self._w2v.vector_size

    def _jtypes2int(self, jtypes):
        return [self._jtype2int_dict[jtype] if jtype in self._jtype2int_dict else self._jtype2int_dict[unk_tok] for
                jtype in jtypes]

    def _tok2int(self, tokens):
        return [self._vocab2int[tok] if tok in self._vocab2int else self._vocab2int[unk_tok] for tok in tokens]

    def _int2tok(self, ints):
        return [self._int2vocab[item] for item in ints]

    def _char2int(self, token):
        return [self._char_vocab2int[ch] if ch in self._char_vocab2int else self._char_vocab2int[unk_tok] for ch in
                token]

    def _int2char(self, ints):
        return [self._char_int2vocab[i] for i in ints]

    def get_vectors(self):
        all_vectors = self._w2v.vectors
        return all_vectors

    def _get_padded_toks(self, all_tokenized_texts):
        sequences = [self._tok2int(tokens) for tokens in all_tokenized_texts]
        all_padded_toks = pad_sequences(sequences, maxlen=self._max_len, dtype=dtype_int,
                                        padding='post',
                                        truncating='post',
                                        value=self._vocab2int[pad_tok])
        return all_padded_toks

    def _get_padded_jtypes(self, all_jtypes, sizes):
        sequences = [self._jtypes2int([jtypes]) * size for jtypes, size in zip(all_jtypes, sizes)]
        all_padded_jtypes = pad_sequences(sequences, maxlen=self._max_len, dtype=dtype_int,
                                          padding='post',
                                          truncating='post',
                                          value=self._jtype2int_dict[pad_tok])
        return all_padded_jtypes

    def _get_padded_chars(self, all_tokenized_texts):
        all_char_seqs = []

        for text in all_tokenized_texts:
            tokens_seq = [self._char2int(tok) for tok in text] + [[self._char_vocab2int[pad_tok]]] * (
                    self._max_len - len(text))
            tokens_seq = tokens_seq[:self._max_len]
            _temp = pad_sequences(tokens_seq, maxlen=self._max_char_len, dtype=dtype_int,
                                  padding='post',
                                  truncating='post',
                                  value=self._char_vocab2int[pad_tok])
            all_char_seqs.append(_temp)
        all_char_seqs = np.array(all_char_seqs)
        return all_char_seqs

    def _get_outputs(self, all_tokenized_texts):
        all_outputs = []
        sequences = [self._tok2int(tokens[1:] + [end_tok]) for tokens in all_tokenized_texts]
        _temp = pad_sequences(sequences, maxlen=self._max_len, dtype=dtype_int,
                              padding='post',
                              truncating='post',
                              value=self._vocab2int[pad_tok])
        for seq in _temp:
            _t = to_categorical(seq, num_classes=len(list(self._vocab2int.values())))
            all_outputs.append(_t)
        all_outputs = np.array(all_outputs)
        return all_outputs

    def get_inputs_outputs(self, texts, jtypes):
        all_tokenized_texts = [self.extract_tokens(text) for text in texts]
        all_paded_toks = self._get_padded_toks(all_tokenized_texts)
        all_paded_chars = self._get_padded_chars(all_tokenized_texts)
        all_jtypes = self._get_padded_jtypes(jtypes, [len(item) for item in all_tokenized_texts])
        all_outputs = self._get_outputs(all_tokenized_texts)

        return all_paded_toks, all_paded_chars, all_jtypes, all_outputs

    def save(self, name):
        path = path_to_processor.joinpath(name)
        path.mkdir(parents=True, exist_ok=True)
        obj_dict = {}
        for name, obj in self.__dict__.items():
            if name == '_w2v':
                continue
            obj_dict[name] = obj

        with open(path.joinpath('objects.pkl'), 'wb') as f:
            pickle.dump(obj_dict, f)

        self._w2v.save_word2vec_format(path.joinpath('w2v'))
        logger.info('object saved successfully')

    def load(self, name):
        path = path_to_processor.joinpath(name)
        if not path.is_dir():
            logger.error('data processor object for {} does not exists'.format(name))

        with open(path.joinpath('objects.pkl'), 'rb') as f:
            obj_dict = pickle.load(f)

        for name, obj in obj_dict.items():
            self.__setattr__(name, obj)
            logger.info('loaded attribute {} successfully'.format(name))

        w2v = KeyedVectors.load_word2vec_format(path.joinpath('w2v'), binary=False)
        self.__setattr__('_w2v', w2v)
        logger.info('loaded attribute {} successfully'.format('_w2v'))

