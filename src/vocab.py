import logging
import pickle
from collections import Counter

import numpy as np
from transformers import BasicTokenizer

basic_tokenizer = BasicTokenizer()

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


class Vocab():
    def __init__(self, file):
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']

        self._id2label = []
        self.target_names = []
        self.label_weights = []

        self.build_vocab(file)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)

        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    def build_vocab(self, file):
        file = open(file, 'rb')
        data = pickle.load(file)

        self.word_counter = Counter()

        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1

        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        label2name = {0: '科技',
                      1: '股票',
                      2: '体育',
                      3: '娱乐',
                      4: '时政',
                      5: '社会',
                      6: '教育',
                      7: '财经',
                      8: '家居',
                      9: '游戏',
                      10: '房产',
                      11: '时尚',
                      12: '彩票',
                      13: '星座'}

        self.label_counter = Counter(data['label'])

        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])
            self.label_weights.append(count)

    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1

        embeddings[self.unk] = embeddings[self.unk] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        assert len(set(self._id2extword)) == len(self._id2extword)

        return embeddings

    def dump(self, filename):
        f_out = open(filename, 'w')
        num = 0
        for word in self._id2word:
            f_out.write(word + '\n')
            num += 1

        assert num == self.word_size
        logging.info('Dump vocab to %s, total %d words' % (filename, num))

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)
