class LanguageModel(object):
    def __init__(self,
                 word_embedding_size=None,
                 char_embedding_size=None,
                 word_inp_mask_val=None,
                 char_inp_mask_val=None,
                 word_vocab_size=None,
                 char_vocab_size=None,
                 max_seq_len=None,
                 max_word_len=None,
                 char_cnn_filters=None,
                 char_cnn_ker_size=None,
                 char_cnn_pool_size=None,
                 lstm_size=()
                 ):
        self._model = None