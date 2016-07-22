import os
import pickle
import numpy as np


def load_embedding(path, name):
    dict_filename = os.path.join(path, name + '.dict.pkl')
    embd_filename = os.path.join(path, name + '.embd.npy')
    if not os.path.exists(dict_filename) or not os.path.exists(embd_filename):
        print('Can not found pickled embedding file.')
        return None, None
    with open(dict_filename, 'rb') as d_f, open(embd_filename, 'rb') as n_f:
        print('Loading pickled embedding.')
        word_dict = pickle.load(d_f)
        embedding = np.load(n_f)
    return word_dict, embedding


if __name__ == '__main__':
    import config

    word_dict, embd = load_embedding(config.GLOVE_PATH, config.GLOVE_NAME)
    print(len(word_dict))
    print(embd.shape)