import fasttext
import sys
fold = sys.argv[1]
print('fold: ', fold)

train = "train_" + str(fold)
dev = "dev_" + str(fold)

train_file = './data/' + train + '.fasttext.txt'
model = fasttext.train_supervised(input=train_file)
model.save_model('./save/fasttext/' + str(fold) + '.bin')

dev_file = './data/' + dev + '.fasttext.txt'
model.test(dev_file)

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

print_results(*model.test(dev_file))