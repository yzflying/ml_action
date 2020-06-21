import numpy as np


def classify0(inX, dataSet, labels, k):
    """
    KNN邻近算法，分类器
    inX(1Xn):输入待测试n维数据,此处只预测一个样本； dataSet(mXn)：输入学习n维数据集X； labels(mX1)：输出分类标签； k：选取距inX邻近的k个学习数据集；
    本函数计算inX的的输出分类标签
    """
    """计算与数据集的距离并排序"""
    dataSetsize = dataSet.shape[0]    #计算数据集的行数
    inXX = np.array((inX*dataSetsize)).reshape(dataSet.shape)   #对inX转化为（4*2）的array       #测试数据为矩阵时，不能这么写
    print('inXX is like:', inXX)

    diffMat = dataSet - inXX                           #差矩阵
    print('diffMat is like:', diffMat)

    distances = (((diffMat)**2).sum(axis=1))**0.5    #差矩阵的平方和(按行)、再开方，得到距离矩阵
    print('distances is like:', distances)

    sort_distances_index = distances.argsort()  #对距离矩阵排序后得到的的索引

    """查找最短的k个距离对应数据集的标签，并统计各标签次数"""
    class_count = {}
    for i in range(k):
        votelabel = labels[sort_distances_index[i]]   #选择距离由近到远的学习数据对应label
        class_count[votelabel] = class_count.setdefault(votelabel, 0) + 1  #创建dict{标签：次数}
    print('class_count is like:', class_count)

    sort_class_count = sorted(class_count.items(), key=lambda item: item[1], reverse=True)   #对class_count按值倒序,每个item对应字典元素的元组，即('B', 2)

    """返回次数最多的标签"""
    print('sort_class_count is like:', sort_class_count)          # [('B', 2), ('A', 1)]
    return sort_class_count[0][0]   #返回count次数最多的label


if __name__ == "__main__":
    dataSet = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])   #输入的学习二维数据集
    labels = ['A', 'A', 'B', 'B']                                  #输出的分类标签
    inX = [0, 0]
    k = 3
    print(classify0(inX, dataSet, labels, k))


"""
打印如下：
inXX is like: [[0 0]
 [0 0]
 [0 0]
 [0 0]]
diffMat is like: [[1.  1.1]
 [1.  1. ]
 [0.  0. ]
 [0.  0.1]]
distances is like: [1.48660687 1.41421356 0.         0.1       ]
class_count is like: {'B': 2, 'A': 1}
sort_class_count is like: [('B', 2), ('A', 1)]
B
"""
