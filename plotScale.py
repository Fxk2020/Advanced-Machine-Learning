# -*- encoding: utf-8 -*-
'''
@File    :   plotScale.py   
@Contact :   13105350231@163.com
@License :   (C)Copyright 2022-2025
@Desciption : 

@Modify Time      @Author    @Version   
------------      -------    --------   
2022/11/24 16:48   fxk        1.0         
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rc('font', family='FangSong', weight='bold')


def plotData(a, b, c):
    """
        画出训练集中各个情感所占的比例
        :param a: 正面情感
        :param b: 中性情感
        :param c: 负面情感
        :return:
        """
    y = np.array([a, b, c])

    plt.pie(y,
            labels=['正面情感', '中性情感', '负面情感'],  # 设置饼图标签
            colors=["#d5695d", "#5d8ca8", "#65a479"],  # 设置饼图颜色
            explode=(0, 0, 0),  # 0.1第二部分突出显示，值越大，距离中心越远
            autopct='%.2f%%',  # 格式化输出百分比
            )
    plt.title("验证集中情感所占比例")
    plt.show()


def getABC(str):
    """
        获得训练集中各个情感的数目
        :param str: 训练集路径
        :return:
        """
    a = 0
    b = 0
    c = 0
    # sep 指定分隔符
    data = pd.read_table(str, sep=",")
    for i in range(len(data.values)):
        if data.values[i][2] == 0:
            a = a + 1
        if data.values[i][2] == 1:
            b = b + 1
        if data.values[i][2] == 2:
            c = c + 1
    # print(data.values[0][2])
    return a, b, c


if __name__ == '__main__':
    a, b, c = getABC("torchtextfiles/val.csv")  # torchtextfiles/val.csv  torchtextfiles/train.csv
    # print(a)
    # print(b)
    # print(c)
    plotData(a, b, c)
