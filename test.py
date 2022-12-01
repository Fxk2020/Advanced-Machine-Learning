import torch
import torch.cuda
import datetime

from main import getVocab
from net import Net
from torchsummary import summary

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # for i in range(5000000):
    #     print(i)
    print(torch.__version__)

    print(torch.cuda.is_available())
    print(1e-3)
    # print(torchtext.__version__)
    #
    # use_gpu = torch.cuda.is_available()
    # print(use_gpu)
    #
    # import visdom
    # import numpy as np
    # vis = visdom.Visdom()
    # vis.text('Hello, world!')
    # vis.image(np.ones((3, 10, 10)))
    IDX, TEXT, _, _, _ = getVocab()
    # 词向量测试用例
    print(type(TEXT))
    print(type(TEXT.vocab))
    print(TEXT.vocab.itos[:200])  # 显示前200个词语
    # print(TEXT.vocab.vectors[98583])  # 显示'德福'的词向量
    print(len(TEXT.vocab))

    # long running
    # do something other
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = Net(len(TEXT.vocab), 200, 256).to(device)
    print(t)
    summary(t)


