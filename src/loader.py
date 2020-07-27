import logging
import pickle
from pathlib import Path

from module import *


def get_examples(file_name, word_encoder, vocab, max_sent_len=256, max_segment=8):
    mode = file_name.split('/')[-1][:-7]  # train validation test

    if isinstance(word_encoder, WordBertEncoder):
        cache_name = './data/' + mode + '.bert.pickle'
    else:
        cache_name = './data/' + mode + '.lstm.pickle'

    if Path(cache_name).exists():
        file = open(cache_name, 'rb')
        examples = pickle.load(file)
        logging.info('Data from cache file: %s, total %d docs.' % (cache_name, len(examples)))
        return examples

    label2id = vocab.label2id
    examples = []

    file = open(file_name, 'rb')
    data = pickle.load(file)

    for text, label in zip(data['text'], data['label']):
        # label
        id = label2id(label)

        # words
        if isinstance(word_encoder, WordBertEncoder):
            sents_words = sentence_split(text, vocab, max_sent_len - 2, max_segment)
            doc = []
            for _, sent_words in sents_words:
                token_ids = word_encoder.encode(sent_words)
                sent_len = len(token_ids)
                token_type_ids = [0] * sent_len
                doc.append([sent_len, token_ids, token_type_ids])
            examples.append([id, len(doc), doc])
        else:
            sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
            doc = []
            for sent_len, sent_words in sents_words:
                word_ids = vocab.word2id(sent_words)
                extword_ids = vocab.extword2id(sent_words)
                doc.append([sent_len, word_ids, extword_ids])
            examples.append([id, len(doc), doc])

    logging.info('Data from file: %s, total %d docs.' % (file_name, len(examples)))

    file = open(cache_name, 'wb')
    pickle.dump(examples, file)
    logging.info('Cache Data to file: %s, total %d docs.' % (cache_name, len(examples)))
    return examples


def sentence_split(text, vocab, max_sent_len=256, max_segment=16):
    words = text.strip().split()
    document_len = len(words)

    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        segments.append([len(segment), segment])

    assert len(segments) > 0
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        return segments


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield docs


def data_iter(data, batch_size, shuffle=True, noise=1.0):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle:
        np.random.shuffle(data)

    lengths = [example[1] for example in data]
    noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
    sorted_indices = np.argsort(noisy_lengths).tolist()
    sorted_data = [data[i] for i in sorted_indices]

    batched_data.extend(list(batch_slice(sorted_data, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch