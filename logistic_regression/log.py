import numpy as np
import math


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
        labels.append(int(line[-1]))      #注意字符型转换为数字类型
        index = index + 1
    return dataSet, labels


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def gradAscent(dataSet_new, labels):
    """
    梯度上升算法
    输入：完整的输入矩阵X：dataSet_new，分类结果Y：labels
    输出：拟合W矩阵：weights
    """
    dataMatrix = np.mat(dataSet_new)   #输入矩阵X (100X3)
    labelMatrix = np.mat(labels).transpose()   #输出单列矩阵Y (100X1)  转换矩阵，方便后面的计算
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycle = 500
    weights = np.mat(np.ones((n, 1)))          #初始化拟合系数矩阵W(3X1)
    for i in range(maxCycle):
        h = sigmoid(np.dot(dataMatrix, weights))      #根据拟合系数W得到的结果，归一化后的输出(100X3)*(3X1)=(100X1),矩阵点乘和*结果一样
        """ weights:"""
        weights = weights + alpha * (dataMatrix.transpose()) * (labelMatrix - h)   # w迭代（对交叉熵代价函数求导）  #每次迭代需要计算所有输入数据
    return weights


if __name__ == "__main__":
    """生成输入矩阵X为dataSet_new，100X3"""
    dataSet, labels = loadDataset("testSet.txt")    #得到训练矩阵100X2 和分类
    first_column = np.array([1.0]*len(dataSet))
    dataSet_new = np.insert(dataSet, 0, values=first_column, axis=1)   #对输入训练矩阵左侧加上一列1.0，即100X3
    """生成标签，100X1"""
    labels = np.array(labels)         #列表转换数组

    a = gradAscent(dataSet_new, labels)
    print(a)


