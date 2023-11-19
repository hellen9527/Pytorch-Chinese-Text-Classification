# 中文文本分类模型
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

基于pytorch与sklearn的中文文本分类代码集合，包括常见的传统机器学习与深度学习等模型，旨在帮助有需要的人更快上手实操，开箱即用。


## 介绍
实现的模型结构以下几类，具体的模型结构的讲解可以看下我的[博客](https://zhuanlan.zhihu.com/p/67680037)
1. 传统机器学习，如：决策树、随机森林、线性SVM、xgboost
2. 传统深度学习模型，如：TextCNN、TextLSTM、FastText
3. 图神经网络模型，如：TextGCN (开发中)
4. 基于预训练语言模型，如：BERT-CLS

## 环境
- python 3.7  
- pytorch 1.2  
- tqdm  
- jieba
- numpy
- sklearn  
- transformers

## 中文数据集

这里使用的是一份公开的京东的用户评论的数据，类别：好评、差评，任务相对比较简单。为了统一数据格式，所以先用jieba分词器进行了分词，没有用领域词典、过滤停用词等。

数据集划分：

数据集|数据量
--|--
训练集|2.4万
验证集|0.3万
测试集|0.3万

### 更换自己的数据集
 - 可以按我的格式构造数据，主要两个字段：文本(words)、标签(label)，其中words是已经分好词的

## 使用说明
首先下载代码到本地，然后进入根目录，根据想要的模型运行相应的命令
``` bash
git clone 
cd 
# 传统机器学习
bash scripts/run_ml_cls.sh 
# 传统DNN深度学习
bash scripts/run_dnn_cls.sh
# 预训练语言模型
bash scripts/run_plm_cls.sh
# 
```
传统机器学习的启动脚本(run_ml_cls.sh)中, 可以修改model_type参数来切换不同的模型 
- jd, 表示使用决策树模型
- rf, 表示随机森林模型, n_jobs只有这个模型才会真正有多cpu加速
- svm, 线性SVM模型
- xgb, XGBoost模型
- lr，逻辑回归模型

传统DNN模型的启动脚本(run_dnn_cls.sh)中, model_type 可以取值如下，如果样本不均衡，可以考虑使用focal loss或者指定各个类别的loss权重（线性简单的可以设置为另一个类别的个数，出现越少权重越大，可以参考cnn的设置方式）
- cnn, 表示TextCNN模型[1]
- lstm, 表示TextLSTM模型[2]，可以再设置lstm_attention参数为"true"，模型变成lstm_attention
- fasttext, 表示FastText模型[3]

基于预训练语言模型的启动脚本(run_plm_cls.sh), model_type 可以取值：
- bert, 表示使用bert-base-chinese作为编码结构, [cls]进行分类[4]

## 效果
每训练100个step在开发集进行评估，选最好的模型在测试集上评估

模型|acc|备注
--|--|--
决策树|93.40%|DecisionTreeClassifier
随机森林|97.01%|RandomForestClassifier   
SVM|97.20%|LinearSVC  
XGBOOST|95.68%|xgb.XGBClassifier()  
TextCNN|96.46%|参考论文[1] 
TextLSTM|95.71%|参考论文[2] 
FastText|93.76%|参考论文[3] 
BERT-CLS|**98.99%**|参考论文[4] 

上表所有的模型结果都没有调参，所以无法精确的进行比较，但从趋势来看，传统机器学习对于简单的分类任务也一样可以取得和传统DNN类似的效果。另外BERT无需调优（也没什么可以调的），效果确实是最好的（要显卡）

## 更新日志
- 20231112 可以处理json格式数据，传统机器学习、传统DNN模型、BERT模型
- 20231119 增加处理txt格式用\t分隔；样本不均衡时的类别loss权重代码，或者指定focal loss代码；lstm后接attention

## 参考论文
[1] Kim Y. Convolutional neural networks for sentence classification[J]. arXiv preprint arXiv:1408.5882, 2014.

[2] Zhou C, Sun C, Liu Z, et al. A C-LSTM neural network for text classification[J]. arXiv preprint arXiv:1511.08630, 2015.

[3] Joulin A, Grave E, Bojanowski P, et al. Fasttext. zip: Compressing text classification models[J]. arXiv preprint arXiv:1612.03651, 2016.

[4] Reimers N, Gurevych I. Sentence-bert: Sentence embeddings using siamese bert-networks[J]. arXiv preprint arXiv:1908.10084, 2019.

## 其他
- 代码还有很多不完善或者不正确的地方，欢迎大家指出或者共建
- 有志于搭建通用NLP开源平台或者需要NLP相关技术帮助的朋友可以加我1527909546@qq.com
