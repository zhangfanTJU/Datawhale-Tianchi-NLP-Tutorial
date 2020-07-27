# for TextRNN
nohup python train.py --config_file lstm.cfg --w lstm --s lstm --seed 666 > log/lstm.lstm.log 2>&1 &

# for TextCNN
nohup python train.py --config_file cnn.cfg --w cnn --s lstm --seed 666   > log/cnn.lstm.log 2>&1 &

# for BERT
nohup python train.py --config_file bert.cfg --w bert --s lstm --seed 666 > log/bert.lstm.log 2>&1 &