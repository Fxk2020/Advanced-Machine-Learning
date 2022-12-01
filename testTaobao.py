# -*- encoding: utf-8 -*-
'''
@File    :   testTaobao.py   
@Contact :   13105350231@163.com
@License :   (C)Copyright 2022-2025
@Desciption : 

@Modify Time      @Author    @Version   
------------      -------    --------   
2022/11/25 15:34   fxk        1.0         
'''
import numpy as np
import pandas as pd
import torch
import torchtext
from torchtext.legacy import data

from net import Net

BATCH_SIZE = 64  # batch大小
FIX_CHARS = 200  # 句子的最大长度
device = torch.device('cuda')  # 是否GPU加速 cuda CPU


def changeId(str):
    """
    更改id的位置
    :param str:
    :return:
    """
    data = pd.read_csv(str, on_bad_lines='skip')
    print(data.columns)
    order = ['id', 'text', 'label']
    data = data[order]
    print(data)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv('testfiles2/taobao4.csv', index=False)


def addId(str):
    """
    为文件添加id
    :param str:csv文件的路径
    :return:
    """
    data = pd.read_csv(str, on_bad_lines='skip')
    print(data.columns)  # 获取列索引值
    dataAdd = []
    for i in range(len(data)):
        i = i + 1
        dataAdd.append(i)
    data['id'] = dataAdd  # 将新列的名字设置为id
    data.to_csv(str, mode='a', index=False)
    # mode=a，以追加模式写入,header表示列名，默认为true,index表示行名，默认为true，再次写入不需要行名
    # print(data)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv('testfiles2/taobao3.csv', index=False)


def tokenizer_text(text):
    """
    Field的tokenize，见split2batches函数。
    :param text:
    :return:
    """
    return text.split()


def tokenizer_id(text):
    """
    不处理id文本。
    :param text:
    :return:
    """
    return text


def tokenizer_label(id):
    """
    返回int型的id。
    :param id:
    :return:
    """
    return int(id)


def getVocab():
    """
    返回TEXT，里面有词向量的权重等参数。
    :return:
    """

    IDX = data.Field(sequential=False, tokenize=tokenizer_id, use_vocab=True)  # sequential=False就不用拆分该字段内容，保留整体。
    TEXT = data.Field(sequential=True, tokenize=tokenizer_text, use_vocab=True, fix_length=FIX_CHARS)
    LABEL = data.Field(sequential=False, tokenize=tokenizer_label, use_vocab=False, dtype=torch.long)

    train, val, test = data.TabularDataset.splits(path='testfiles2', train='train.csv', validation='val.csv',
                                                  test='taobao4.csv', format='csv', skip_header=True,
                                                  fields=[('id', IDX), ('text', TEXT), ('label', LABEL)])

    vects = torchtext.vocab.Vectors(
        name='wordEmbeddings/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt',
        # name='wordEmbeddings/Sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5.txt',
        cache='cachefiles2/')
    TEXT.build_vocab(train, vectors=vects)

    IDX.build_vocab(test)

    """词向量测试用例
    print(type(TEXT))
    print(type(TEXT.vocab))
    print(TEXT.vocab.itos[:200])  # 显示前200个词语
    print(TEXT.vocab.vectors[98583])  #
    """

    return IDX, TEXT, train, val, test


def split2batches(batch_size=BATCH_SIZE, device='cpu'):
    """
    按批次分割，生成迭代器。
    :param batch_size:
    :param device:
    :return:
    """

    _, _, train, val, test = getVocab()

    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=batch_size,
                                                                 device=device, sort_key=lambda x: len(x.text),
                                                                 sort_within_batch=False)

    return train_iter, val_iter, test_iter


def test(net, test_iter, IDX):
    """
    对官方的测试集进行测试。
    :param net:  加载训练好的LSTM网络模型
    :param test_iter:
    :param IDX:
    :return:
    """
    temp = []
    id = []
    label = []

    realLabel = []

    for step, item in enumerate(test_iter):
        id.extend(item.id)
        item.text = item.text.to(device)  # GPU加速
        realLabel.extend(item.label)
        with torch.no_grad():
            pre = net(item.text)
            pre = pre.argmax(dim=1)  # 示例tensor([0,1,2,1,1,...,])
            pre = pre.cpu().data.numpy()
            pre = pre.tolist()  # 示例[0,1,2,1,1,...,]
            label.extend(pre)

    id = np.array(id)  # array([1,2,5,3,...])
    id = id.tolist()  # [1,2,5,3,...]
    realLabel = np.array(realLabel).tolist()

    for item in id:  # 将id中的数值转换为真实的id文本
        temp.append(IDX.vocab.itos[item])

    # dataframe = pd.DataFrame({'id': temp, 'label': label})
    # dataframe.to_csv('endclass.csv', index=False)

    acc_num = 0
    for i in range(len(realLabel)):
        # print("------------------------------")
        # print(label[i])
        # print(realLabel[i])
        if realLabel[i] == label[i]:
            print(i)
            acc_num = acc_num + 1

    print("准确率是", end=" ")
    print(acc_num / len(realLabel))
    return realLabel, label


if __name__ == '__main__':
    net = Net(168591, 200, 128).to(device)  # embedding_dim=300, hidden_dim=128 # GPU加速
    net.load_state_dict(torch.load('best_par.pt'))
    # 测试所保存的模型
    print('加载最优模型参数完成...')

    # addId('testfiles2/taobao2.csv')
    # changeId('testfiles2/taobao3.csv')

    test_iter = split2batches()
    IDX, TEXT, _, _, _ = getVocab()
    train_iter, val_iter, test_iter = split2batches()
    realLabel, label = test(net, test_iter, IDX)
    print(np.array(realLabel).tolist())
    print(label)
