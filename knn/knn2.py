import numpy as np


def classify0(inX, dataSet, labels, k):
    """
    KNN邻近算法，分类器
    inX:输入待测试n维数据（1Xn）； dataSet：输入学习n维数据集； labels：输出分类标签； k：选取距inX邻近的k个学习数据集；
    本函数计算inX的的输出分类标签
    """
    """计算与数据集的距离并排序"""
    dataSetsize = dataSet.shape[0]    #计算数据集的行数
    inXX = np.tile(inX, (dataSetsize, 1))   #对inX转化为的array，inXX(mXn)          注意inX*dataSetsize中只有inX是list才正确，inX是矩阵时，*dataSetsize不会改变inX的shape
    diffMat = dataSet - inXX                           #差矩阵
    distances = (((diffMat)**2).sum(axis=1))**0.5    #差矩阵的平方和(按行)、再开方，得到距离矩阵
    sort_distances_index = distances.argsort()  #对距离矩阵排序后得到的的索引
    """查找最短的k个距离对应数据集的标签，并统计各标签次数"""
    class_count = {}
    for i in range(k):
        votelabel = labels[sort_distances_index[i]]   #选择距离由近到远的学习数据对应label
        class_count[votelabel] = class_count.setdefault(votelabel, 0) + 1  #创建dict{标签：次数}
    sort_class_count = sorted(class_count.items(), key=lambda item: item[1], reverse=True)   #对class_count按值倒序
    """返回次数最多的标签"""
    return sort_class_count[0][0]   #返回count次数最多的label


def file_to_matrix(filename):     #根据filename输出：3维输入数据集dataSet，输出分类标签labels
    """
    :param filename: 数据集文件名称
    :return:dataSet：样本集(mX3)，np.array类型；labels：样本标签，list类型
    """
    data = open(filename)
    arraylines = data.readlines()  #按行读取，arraylines为list变量，长度为行数
    m = len(arraylines)
    dataSet = np.zeros((m, 3))     #创造0矩阵，存储3维输入数据集
    labels = []
    index = 0
    for line in arraylines:
        line = line.split()       #line为该行的元素列表
        dataSet[index, :] = line[0:3]
        labels.append(line[-1])
        index = index + 1
    return dataSet, labels


def autoNorm(dataSet):      #归一化特征值（输入数据集dataSet），将每个特征值减去最小值再除以变化范围，以减小特征之间的差异对结果影响
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normdataSet = (dataSet - np.tile(minVals, (m, 1)))/(np.tile(ranges, (m, 1)))      #np.tile(ranges, (m, 1)) 把ranges矩阵横着写1遍，再竖着写m遍
    return normdataSet


if __name__ == "__main__":
    k = 3
    filename = 'datingTestSet.txt'
    dataSet, labels = file_to_matrix(filename)
    normdataSet = autoNorm(dataSet)
    m = normdataSet.shape[0]    # 即样本数量
    num_test = int(m*0.1)
    cnt = 0
    """前num_test，即前1/10的数据集为测试集，后9/10的数据集为训练集"""
    for i in range(num_test):
        test_label = classify0(normdataSet[i, :], normdataSet[num_test:m, :], labels[num_test:m], k)              #注意取子矩阵要用两个参数（，后不可省略）
        if test_label == labels[i]:
            cnt = cnt + 1
            print("YES!!! test is %s, and right is %s" % (test_label, labels[i]))
        else:
            print("NO!!! test is %s, and right is %s" % (test_label, labels[i]))
    print("test number is %d, and right number is %d, right rate is %f!" % (num_test, cnt, (cnt/num_test)))

