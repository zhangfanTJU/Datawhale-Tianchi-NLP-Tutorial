import logging
import os
import random
from configparser import ConfigParser

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


class Config(object):
    def __init__(self, args):
        self.word_encoder = args.w
        self.sent_encoder = args.s
        self.gpu_id = args.gpu

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)

        config = ConfigParser()
        config.read('./config/' + args.config_file)

        self._config = config
        self.train_file = './data/train_' + str(args.fold) + '.pickle'
        self.dev_file = './data/dev_' + str(args.fold) + '.pickle'

        _config = {}
        for section in config.sections():
            for k, v in config.items(section):
                _config[k] = v
        import json
        logging.info(json.dumps(_config, indent=1))

        self.save_dir = './save/' + args.w + '.' + args.s + '/' + str(args.fold)

        self.log_dir = self.save_dir
        self.save_model = self.save_dir + '/module.bin'
        self.save_config = self.save_dir + '/config.cfg'
        self.save_test = self.save_dir + '/test.res'

        if self.save and not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
            config.write(open(self.save_config, 'w'))

        logging.info('Load config form ' + args.config_file)

    @property
    def word2vec_path(self):
        return self._config.get('Data', 'word2vec_path')

    @property
    def bert_path(self):
        return self._config.get('Data', 'bert_path')

    @property
    def save(self):
        return self._config.getboolean('Save', 'save')

    @property
    def word_dims(self):
        return self._config.getint('Network', 'word_dims')

    @property
    def dropout_embed(self):
        return self._config.getfloat('Network', 'dropout_embed')

    @property
    def dropout_mlp(self):
        return self._config.getfloat('Network', 'dropout_mlp')

    @property
    def word_num_layers(self):
        return self._config.getint('Network', 'word_num_layers')

    @property
    def word_hidden_size(self):
        return self._config.getint('Network', 'word_hidden_size')

    @property
    def sent_num_layers(self):
        return self._config.getint('Network', 'sent_num_layers')

    @property
    def sent_hidden_size(self):
        return self._config.getint('Network', 'sent_hidden_size')

    @property
    def dropout_input(self):
        return self._config.getfloat('Network', 'dropout_input')

    @property
    def dropout_hidden(self):
        return self._config.getfloat('Network', 'dropout_hidden')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def bert_lr(self):
        return self._config.getfloat('Optimizer', 'bert_lr')

    @property
    def decay(self):
        return self._config.getfloat('Optimizer', 'decay')

    @property
    def decay_steps(self):
        return self._config.getint('Optimizer', 'decay_steps')

    @property
    def beta_1(self):
        return self._config.getfloat('Optimizer', 'beta_1')

    @property
    def beta_2(self):
        return self._config.getfloat('Optimizer', 'beta_2')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def clip(self):
        return self._config.getfloat('Optimizer', 'clip')

    @property
    def gamma(self):
        return self._config.getint('Optimizer', 'gamma')

    @property
    def threads(self):
        return self._config.getint('Run', 'threads')

    @property
    def epochs(self):
        return self._config.getint('Run', 'epochs')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def log_interval(self):
        return self._config.getint('Run', 'log_interval')

    @property
    def early_stops(self):
        return self._config.getint('Run', 'early_stops')

    @property
    def save_after(self):
        return self._config.getint('Run', 'save_after')

    @property
    def update_every(self):
        return self._config.getint('Run', 'update_every')
