# 零基础入门NLP-新闻文本分类

* [天池比赛地址](https://tianchi.aliyun.com/competition/entrance/531810/introduction?spm=5176.12281973.1005.1.65531f54eHuMAA)
* [Datawhale开源地址](https://github.com/datawhalechina/team-learning-nlp/tree/master/NewsTextClassification)

## 模型
* TextCNN
* TextRNN
* HAN
* BERT

## 目录
```
├── bert                # BERT预训练  
├── config              # 模型配置    
├── data                # 训练数据   
├── docs                # 教程文档   
├── emb                 # 词向量以及BERT权重    
├── module              # 模块相关代码    
├── src                 # 训练相关代码   
├── preprocessing.py    # 预处理   
├── README.md           # 说明文档  
├── train.py            # 训练代码  
├── train.sh            # 训练脚本  
└── word2vec.py         # word2vec训练代码  
```

# 依赖
* [fitlog](https://github.com/fastnlp/fitlog)
* gensim
* pandas
* pytorch == 1.2.0
* transformers == 2.9.0
* tensorflow == 1.12

# 快速开始
1. 将数据和词向量分别放在`data`和`emb`目录下,初始化fitlog `fitlog init .`
2. 运行预处理代码`python preprocessing.py`
3. 运行训练脚本`bash train.sh`


## 关于Datawhale

> Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale 以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时 Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。

欢迎关注：

 ![](http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/1095279172547/1584432602983_kAxAvgQpG2.jpg)