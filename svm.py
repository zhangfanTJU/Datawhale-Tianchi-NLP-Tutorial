import logging
import pickle
import time
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

from src.utils import *
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--fold', default=9, type=int, help='fold for test')
args = argparser.parse_args()

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

# train data
model_file = './res/svm/svm_' + str(args.fold) + '.model'
save_file = './res/svm/dev_' + str(args.fold)

file = open('./data/train_' + str(args.fold) + '.tfidf.pickle', 'rb')
train_data = pickle.load(file)
logging.info('| {} | features {}'.format('train data ' + str(args.fold), train_data['text'].shape))

# model
start_time = time.time()

if Path(model_file).exists():
    # load
    file = open(model_file, 'rb')
    model = pickle.load(file)
    logging.info('| {} | times {:.2f}s'.format('load model', time.time() - start_time))
else:
    # train
    model = SVC(C=1.0, kernel="linear")
    model.fit(train_data['text'], train_data['label'])

    logging.info('| {} | times {:.2f}s'.format('train model', time.time() - start_time))

    # res
    file = open(model_file, 'wb')
    pickle.dump(model, file)
    file.close()
    logging.info('res model.')

# predict train
start_time = time.time()

y_pred_train = model.predict(train_data['text'])
score, score_no_0, f1_no_0 = get_score(train_data['label'], y_pred_train)
logging.info('| {} | score {} | score_no_0 {} | {}'.format('train', score, score_no_0, f1_no_0))
report = classification_report(train_data['label'], y_pred_train, digits=4, target_names=list(label2name.values()))
logging.info('\n' + report)

logging.info('| {} | times {:.2f}s'.format('train', time.time() - start_time))

# predict dev
file = open('./data/dev_' + str(args.fold) + '.tfidf.pickle', 'rb')
dev_data = pickle.load(file)
logging.info('| {} | features {}'.format('dev data ' + str(args.fold), dev_data['text'].shape))

start_time = time.time()

y_pred_dev = model.predict(dev_data['text'])
score, score_no_0, f1_no_0 = get_score(dev_data['label'], y_pred_dev)
logging.info('| {} | score {} | score_no_0 {} | {}'.format('dev', score, score_no_0, f1_no_0))
report = classification_report(dev_data['label'], y_pred_dev, digits=4, target_names=list(label2name.values()))
logging.info('\n' + report)

# res
file = open(save_file, 'w')
file.write(str(list(y_pred_dev)))
file.write('\n')
file.write(str(dev_data['label']))
file.close()
logging.info('| Save res to {}'.format(save_file))

logging.info('| {} | times {:.2f}s'.format('dev', time.time() - start_time))
