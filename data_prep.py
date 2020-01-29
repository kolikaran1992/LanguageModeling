from gensim.models import KeyedVectors
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import math
from keras.utils import Sequence

tokenizer = RegexpTokenizer(r'\w+|\n+|[^\w\s]')
lower = False

dtype_int = 'int32'
dtype_float = 'float32'


def extract_tokens(text):
    return [tok for tok in tokenizer.tokenize(text)]


pad_tok = '<pad>'
unk_tok = '<unk>'
end_tok = '<end>'


class TextProcessor(object):
    def __init__(self,
                 word2vec_path='',
                 max_len=50,
                 max_char_len=15):
        self._max_len = max_len
        self._max_char_len = max_char_len

        self._w2v = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
        self._vocab2int = {tok: i for i, tok in
                           enumerate([pad_tok] + list(self._w2v.vocab.keys()) + [unk_tok, end_tok])}
        self._int2vocab = {i: tok for i, tok in
                           enumerate([pad_tok] + list(self._w2v.vocab.keys()) + [unk_tok, end_tok])}
        self._char_vocab2int = {ch: idx for idx, ch in enumerate(
            [pad_tok] + list(set([ch for tok in list(self._vocab2int.keys())[1:-1] for ch in tok])) + [unk_tok])}
        self._char_int2vocab = {idx: ch for idx, ch in enumerate(
            [pad_tok] + list(set([ch for tok in list(self._vocab2int.keys())[1:-1] for ch in tok])) + [unk_tok])}

    def get_vocab2int(self):
        return self._vocab2int

    def get_char_vocab2int(self):
        return self._char_vocab2int

    def get_vector_dim(self):
        return self._w2v.vector_size

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
        all_vectors = np.append(np.zeros((1, self._w2v.vector_size)), all_vectors, axis=0)
        all_vectors = np.append(all_vectors, np.random.random((1, self._w2v.vector_size)), axis=0)
        all_vectors = np.append(all_vectors, np.random.random((1, self._w2v.vector_size)), axis=0)
        return all_vectors

    def _get_padded_toks(self, all_tokenized_texts):
        sequences = [self._tok2int(tokens) for tokens in all_tokenized_texts]
        all_padded_toks = pad_sequences(sequences, maxlen=self._max_len, dtype=dtype_int,
                                        padding='post',
                                        truncating='post',
                                        value=self._vocab2int[pad_tok])
        return all_padded_toks

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

    def get_inputs_outputs(self, texts):
        all_tokenized_texts = [extract_tokens(text) for text in texts]
        all_paded_toks = self._get_padded_toks(all_tokenized_texts)
        all_paded_chars = self._get_padded_chars(all_tokenized_texts)
        all_outputs = self._get_outputs(all_tokenized_texts)

        return all_paded_toks, all_paded_chars, all_outputs


class BatchGenerator(Sequence):
    def __init__(self, all_texts, text_processor=None, shuffle=True, batch_size=32):
        self.all_texts = all_texts
        self.batch_size = batch_size
        self._text_processor = text_processor
        self.shuffle = shuffle

    def __getitem__(self, idx):
        batch_texts = self.all_texts[idx * self.batch_size: (idx + 1) * self.batch_size]
        toks, chars, outs = self._text_processor.get_inputs_outputs(batch_texts)
        return [toks, chars], outs

    def __len__(self):
        return math.ceil(len(self.all_texts) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.all_texts)
