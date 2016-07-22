from datetime import datetime
from model.snli_seq_model import SnliBasicLSTM
import config
from utils.load_embd import load_embedding
from expr.expr_wrapper import wrapper
import argparse

if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser(description='Test for some variant of LSTM model.')
    arg_parse.add_argument('-m', '--message', metavar='message', dest='message', action='store',
                           help='Specify a message for this test')

    args = arg_parse.parse_args()

    max_length = 80

    word2id, word_embedding = load_embedding(config.GLOVE_PATH, config.GLOVE_NAME)

    timestamp = '{0:(%Y-%m-%d-%H:%M:%S)}'.format(datetime.now())

    basicLSTM = SnliBasicLSTM(lstm_step=max_length, embedding=word_embedding, hidden_d=100,
                              Time=timestamp, Message=args.message)

    basicLSTM.setup(embedding=word_embedding)
    wrapper(model_name='snli-basicLSTM', model=basicLSTM, max_length=max_length, benchmark=0.70)