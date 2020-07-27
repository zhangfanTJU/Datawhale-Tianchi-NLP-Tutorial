from gensim.models.word2vec import Word2Vec

num_features = 100     # Word vector dimensionality
num_workers = 8       # Number of threads to run in parallel

f = open('./data/train_9.word2vec.txt', 'r')
sentences = f.readlines()
sentences = list(map(lambda x: list(x.split()), sentences))
model = Word2Vec(sentences, workers=num_workers, size=num_features)
model.init_sims(replace=True)

# 保存模型
model.save("./save/word2vec.bin")


# 模型加载
model = Word2Vec.load("./save/word2vec.bin")

# 改变格式
model.wv.save_word2vec_format('./emb/word2vec.txt', binary=False)