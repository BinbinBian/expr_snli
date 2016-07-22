from config import ROOT_DIR
import os
from datetime import datetime
import time
import tensorflow as tf
import json
import logging
import inspect


def build_logger(filename=None, name='reg'):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_handler.setFormatter(formatter)
    logger.addHandler(s_handler)

    logger.setLevel(logging.INFO)

    return logger


class ResultSaver:
    def __init__(self, model_name, model=None, savePara=False, sess=None):
        OUT_ROOT = os.path.join(ROOT_DIR, 'runs')
        timestamp = '{0:(%Y-%m-%d-%H:%M:%S)}'.format(datetime.now())
        self.model = model
        self.saver_root = os.path.abspath(os.path.join(OUT_ROOT, '-'.join(['result', model_name, timestamp])))
        self.checkpoint_dir = os.path.abspath(os.path.join(self.saver_root, "checkpoints"))
        self.meta_filename = os.path.abspath(os.path.join(self.saver_root, "meta.json"))
        self.model_src_filename = os.path.abspath(os.path.join(self.saver_root, "model_src.py"))

        self.global_logger = None
        self.epoch_logger = None

        self.log_print = build_logger()

        self.savePara = savePara
        self.sess = sess
        self.tf_saver = None

    def setup(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        with open(self.meta_filename, 'w', encoding='utf-8') as meta_f:
            json.dump(obj=self.model.model_info, sort_keys=True, indent=4, fp=meta_f)
            print("Saved model meta-info to {}".format(self.meta_filename))
        with open(self.model_src_filename, 'w', encoding='utf-8') as src_f:
            model_src = inspect.getsource(type(self.model))
            src_f.write(model_src)
            src_f.flush()

        global_filename = os.path.abspath(os.path.join(self.saver_root, "best_log.txt"))
        epoch_stat_filename = os.path.abspath(os.path.join(self.saver_root, "epoch_stat.txt"))
        self.global_logger = build_logger(global_filename, 'global')
        self.epoch_logger = build_logger(epoch_stat_filename, 'epoch')

        if self.savePara:
            self.tf_saver = tf.train.Saver(tf.all_variables())

    def logging_best(self, msg):
        if self.savePara:
            ckp_file = self.save_params()
            self.global_logger(msg + ' - ' + ckp_file)
        else:
            self.global_logger.info(msg)

    def logging_epoch(self, msg):
        self.epoch_logger.info(msg)

    def save_params(self):
        timestamp = str(int(time.time())) + '.ckpt'
        path = self.tf_saver.save(self.model.sess, os.path.join(self.checkpoint_dir, timestamp))
        pathinfo = ' '.join(['Path:', path])
        return pathinfo

    def close(self):
        pass
