import os
from pprint import pprint

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

GLOVE_PATH = os.path.join(DATA_DIR, 'glove/')
GLOVE_NAME = 'glove.840B.300d.fine_selected'

SNLI_DEV_FILE = os.path.join(DATA_DIR, 'snli/dev_data.h5')
SNLI_TEST_FILE = os.path.join(DATA_DIR, 'snli/test_data.h5')
SNLI_TRAIN_FILE = os.path.join(DATA_DIR, 'snli/train_data.h5')

SNLI_ST_DEV_FILE = os.path.join(DATA_DIR, 'snli/skipth_dev_data.h5')
SNLI_ST_TEST_FILE = os.path.join(DATA_DIR, 'snli/skipth_test_data.h5')
SNLI_ST_TRAIN_FILE = os.path.join(DATA_DIR, 'snli/skipth_train_data.h5')

SLURM_DATA_ROOT = '/share/data/ripl'

SNLI_ST_DEV_FILE_ON_SLURM = os.path.join(SLURM_DATA_ROOT, 'snli_skipth_data/snli_dev_data.h5')
SNLI_ST_TEST_FILE_ON_SLURM = os.path.join(SLURM_DATA_ROOT, 'snli_skipth_data/snli_test_data.h5')
SNLI_ST_TRAIN_FILE_ON_SLURM = os.path.join(SLURM_DATA_ROOT, 'snli_skipth_data/snli_train_data.h5')

if __name__ == '__main__':
    pprint(locals())