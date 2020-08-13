import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from src.vocab import Vocab


# set seed
random.seed(666)
np.random.seed(666)


def write_data(file_name, data):
    file = open(file_name, 'wb')
    pickle.dump(data, file, protocol=4)
    file.close()
    print(file_name, len(data['text']))


def convert_data_word2vec(filename):
    f = open('./data/' + filename + '.pickle', 'rb')
    data = pickle.load(f)
    num = 0
    f_out = open('./data/' + filename + '.word2vec.txt', 'w')
    for text in data['text']:
        f_out.write(text + '\n')
        num += 1

    assert num == len(data['label'])
    print(filename, num)


def convert_data_bert_pretrain(filename):
    f = open('./data/' + filename + '.pickle', 'rb')
    data = pickle.load(f)
    num = 0
    f_out = open('./data/' + filename + '.bert.txt', 'w')
    for text in data['text']:
        f_out.write(text + '\n')
        f_out.write('\n')
        num += 1

    assert num == len(data['label'])
    print(filename, num)

def all_data2fold(fold_num):
    f = pd.read_csv('./data/train_set.csv', sep='\t', encoding='UTF-8')
    texts = f['text'].tolist()
    labels = f['label'].tolist()

    total = len(labels)

    index = list(range(total))
    np.random.shuffle(index)

    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        print(label, len(data))
        batch_size = int(len(data) / fold_num)
        other = len(data) - batch_size * fold_num
        used = 0
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            batch_data = [data[used + b] for b in range(cur_batch_size)]
            used += cur_batch_size
            all_index[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)

        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}

        file_name = './data/fold_' + str(fold) + '.pickle'
        write_data(file_name, data)


def fold2data(fold_num):
    fold_lens = []
    writer = pd.ExcelWriter('collect.xlsx')
    for fold in range(9, fold_num):
        # dev
        fold_ = fold
        f_dev = open('./data/fold_' + str(fold_) + '.pickle', 'rb')
        dev_data = pickle.load(f_dev)
        file_name = './data/dev_' + str(fold) + '.pickle'
        write_data(file_name, dev_data)

        # train
        train_texts = []
        train_labels = []
        folds = []
        for i in range(1, fold_num):
            fold_ = (fold + i) % fold_num
            folds.append(fold_)
            f_train = open('./data/fold_' + str(fold_) + '.pickle', 'rb')
            data = pickle.load(f_train)
            train_texts.extend(data['text'])
            train_labels.extend(data['label'])

        # collect length
        from collections import Counter
        len_counter = Counter()
        for text in train_texts:
            len_ = int(len(text.split()) / 510)
            len_counter[len_] += 1

        lens, lens_count = [], []
        for len_, count in len_counter.most_common():
            lens.append(len_)
            lens_count.append(count)

        # collect label
        label_counter = Counter()
        for label in train_labels:
            label_counter[label] += 1
        labels, labels_count = [], []
        for label, count in label_counter.most_common():
            labels.append(label)
            labels_count.append(count)

        # write to excel
        len_data = {'lens': lens, 'lens_count':lens_count}
        data_df = pd.DataFrame(len_data)
        data_df.to_excel(writer, sheet_name='lens_'+str(fold), index=False)

        laben_data = {'labels': labels, 'labels_count': labels_count}
        data_df = pd.DataFrame(laben_data)
        data_df.to_excel(writer, sheet_name='labels_'+str(fold), index=False)

        train_data = {'label': train_labels, 'text': train_texts}
        train_name = './data/train_' + str(fold) + '.pickle'
        write_data(train_name, train_data)

        fold_lens.append(str([fold, len(train_data['text']), len(dev_data['text'])])[1:-1])
        print()

    writer.save()

    for fold in range(fold_num):
        print(fold_lens[fold])


if __name__ == "__main__":
    fold_num = 10

    # split data to 10 fold
    all_data2fold(fold_num)

    # fold to train, dev data
    fold2data(fold_num)

    # convert each fold data
    for fold in range(9, fold_num):
        cache_name = "./save/vocab/" + str(fold) + ".pickle"
        train = "train_" + str(fold)
        dev = "dev_" + str(fold)
        files = [train, dev]

        # biuld vocab
        if Path(cache_name).exists():
            vocab_file = open(cache_name, 'rb')
            vocab = pickle.load(vocab_file)
            vocab_name = "./save/vocab/vocab.txt"
            vocab.dump(vocab_name)
            print('Load vocab from ' + cache_name)
        else:
            vocab = Vocab('./data/' + train + '.pickle')
            file = open(cache_name, 'wb')
            pickle.dump(vocab, file)
            print('Save vocab to ' + cache_name)

        for file in files:
            pass

            # data 2 word2vec
            # convert_data_word2vec(file)

            # data 2 bert
            # convert_data_bert_pretrain(train)
