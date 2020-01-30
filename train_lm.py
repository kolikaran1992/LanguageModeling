from argparse import ArgumentParser
from pathlib import Path
import json
from .__logger__ import LOGGER_NAME
import logging
from .training import Training
from keras.optimizers import rmsprop
import sys

sys.path.append('/home/gpuadmin/projects/candice/AGEL/agel_backend/agel_v15/copy_editing/get_grammar_suggestions')

logger = logging.getLogger(LOGGER_NAME + "_console")

parser = ArgumentParser()
parser.add_argument("--params_path", "--params_path",
                    dest="params_path", default={},
                    help="path to parameter values")

args = parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()

    params_path = Path(args.params_path.replace("\\", ''))

    if not params_path.is_file():
        logger.error('{} is not a valid file'.format(params_path.as_posix()))

    with open(params_path, 'r') as f:
        params = json.load(f)

    with open(params['all_jtypes_path'], 'r') as f:
        all_jtype = json.load(f)

    with open(params['path_to_train_data'], 'r') as f:
        data = json.load(f)

    trainer = Training(
        word2vec_path=Path(params['word2vec_path']),
        max_seq_len=params['max_seq_len'],
        max_word_len=params['max_word_len'],
        all_jtypes=all_jtype,
        char_emb_dim=params['char_emb_dim'],
        jtype_emb_size=params['jtype_emb_size'],
        char_cnn_filters=params['char_cnn_filters'],
        char_cnn_ker_size=params['char_cnn_ker_size'],
        char_cnn_pool_size=params['char_cnn_pool_size'],
        random_state=params['random_state'],
        name=params['name'],
        tensorboard_log_path=Path(params['tensorboard_log_path']),
        save_peroid=params['save_peroid']
    )

    opt = rmsprop(lr=0.0001)
    loss = 'categorical_crossentropy'
    trainer.compile_model(optimizer=opt, loss=loss)

    trainer.train(data, batch_size=256, initial_epoch=0, epochs=50)

