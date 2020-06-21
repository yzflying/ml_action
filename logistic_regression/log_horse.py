import numpy as np
import math
import random
import time


def loadDataset(filename):  # 根据filename输出：输入数据集dataSet，输出分类标签labels
    data = open(filename)
    arraylines = data.readlines()  # 按行读取，arraylines为list变量，长度为行数
    m = len(arraylines)
    n = len(arraylines[0].split())
    dataSet = np.zeros((m, n-1))  # 创造0矩阵，存储输入数据集
    labels = []
    index = 0
    for line in arraylines:
        line = line.split()  # line为该行的元素列表
        dataSet[index, :] = line[0:n-1]
        labels.append(float(line[-1]))      #注意字符型转换为数字类型
        index = index + 1
    return dataSet, labels


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def gradAscent(dataSet_new, labels):
    """
    随机梯度上升算法
    输入：完整的输入矩阵X：dataSet_new，分类结果Y：labels
    输出：拟合W矩阵：weights
    """
    dataMatrix = np.mat(dataSet_new)   #输入矩阵X (100X3)
    labelMatrix = np.mat(labels).transpose()   #输出单列矩阵Y (100X1)  转换矩阵，方便后面的计算
    m, n = np.shape(dataMatrix)
    maxCycle = 500
    weights = np.mat(np.ones((n, 1)))          #初始化拟合系数矩阵W(3X1)
    for j in range(maxCycle):                  #迭代次数
        Index_List = [i for i in range(m)]      #每次迭代把所有样本随机用一遍
        for i in range(m):
            alpha = 4/(1.0+i+j) + 0.001        #change1:改变步长，越来越小，减小数据波动
            Cur_Index = Index_List.pop(random.randint(0, len(Index_List)-1))
            h = sigmoid(np.dot(dataMatrix[Cur_Index], weights))      #change2:每次迭代随机选取一个样本，减小数据的随机波动
            weights = weights + alpha * (dataMatrix[Cur_Index].transpose()) * (labelMatrix[Cur_Index] - h)   # w迭代（对代价函数求导） #随机选取样本更新W
    return weights


def classifyVector(inXone, w):          #在test中，根据w和test样本计算得到label（用于和实际label比较）
    prob = sigmoid(np.dot(inXone, w))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


if __name__ == "__main__":
    """加载训练集并分离标签，dataSet（299X21）"""
    dataSet, labels = loadDataset("horseColicTraining.txt")
    """训练得到模型参数W（21X1）"""
    w = gradAscent(dataSet, labels)    #根据样本计算得到拟合系数矩阵w
    print(w)
    """加载测试集并分离标签，dataSet（67X21）"""
    dataSet_test, labels_test = loadDataset("horseColicTest.txt")
    m = np.shape(dataSet_test)[0]     # 待测试样本个数
    error_cnt = 0
    for i in range(m):
        if classifyVector(dataSet_test[i], w) != labels_test[i]:
            error_cnt += 1
    print(error_cnt)




