import numpy as np
import math
import random


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


def selectJrand(i, m):      #随机选择样本i和j
    j = i
    while (j == i):
        j = random.randint(0, m-1)
    return j


def clipAlpha(a, H, L):  #限值
    """alphas矩阵元素的值a不仅要满足0<a<C，还要满足L<a<H"""
    if a > H:
        a = H
    if a < L:
        a = L
    return a


def smoSimple(dataMatin, classLabels, C, toler, maxIter): #数据集；输出类别；C；容错率；最大循环次数
    """
    :param dataMatin: 训练集输入，（100X2）
    :param classLabels: 训练集标签
    :param C: 惩罚因子，定义C>0，在损失函数中的惩罚项的权重
    :param toler:容错率，也叫软间隔，设定toler>=0，保证即使有噪音，也能保证函数间隔y(wx + b) >= 1-toler 恒成立
    :param maxIter:最大循环次数
    :return:alphas、b模型参数
    """
    dataMatrix = np.mat(dataMatin)      # 将输入数据集转换为（100X2）矩阵， m*n
    labelMat = np.mat(classLabels).transpose()  # 将标签转换为100X1的矩阵，m*1
    """初始化模型参数b、alphas"""
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))   #m*1,待优化的alphas向量
    iter = 0

    while(iter < maxIter):
        alphaPairsChanged = 0
        """取消前最大循环次数iter：循环m个样本，如果alphas某个值改变了，alphaPairsChanged不为0，iter置0；
                                    循环m个样本，如果alphas所有值未改变，iter加1,；如果循环maxIter次，每次循环m个样本，alphas未改变，则退出循环，返回alphas、b
                                    即：连续性循环maxIter次，alphas所有值未改变，则退出
        """
        for i in range(m):
            """
            对于m个样本，每次循环步骤如下：
            1.计算样本模拟值fXi，误差Ei；如果误差Ei较大，且0<alphas[i] < C，则可以优化；
                否则退出本循环，计算下一个样本
            """
            fXi = float((np.multiply(alphas, labelMat).transpose()) * dataMatrix * (dataMatrix[i, :].transpose())) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):   # 对alphas[i]对应实例，如果Ei误差较大且alphas[i]有优化余地
                """
                2.如果样本i可以优化，则：
                2.1 随机选取另一个样本j
                """
                j = selectJrand(i, m)    #随机选取J
                """2.2 计算样本j的误差Ej"""
                fXj = float((np.multiply(alphas, labelMat).transpose()) * dataMatrix * (dataMatrix[j, :].transpose())) + b
                Ej = fXj - float(labelMat[j])
                """2.3 alphas[i]、alphas[j]的优化前的值alphaIold、alphaJold"""
                alphaIold = alphas[i].copy()
                alphaJold = alphas[i].copy()
                """2.4 对i，j，计算alphas[j]的约束条件L、H"""
                if (labelMat[i] != labelMat[j]):           # 如果i，j的标签不一致
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[i] - alphas[j])
                else:                                      # 如果i，j的标签一致
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    print('L==H')
                    continue
                """2.5 计算eta"""
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].transpose() - dataMatrix[i, :] * dataMatrix[i, :].transpose() - dataMatrix[j, :] * dataMatrix[j, :].transpose()
                if eta >= 0:
                    print("eta>=0")
                    continue
                """2.6 计算alphas[j]的值，并利用约束条件L、H进行限定"""
                alphas[j] -= labelMat[j] * (Ei-Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                """2.7 计算alphas[i]的值"""
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold-alphas[j])
                """2.8 计算b的值"""
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].transpose() - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].transpose()
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].transpose() - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].transpose()
                if (alphas[i] > 0) and (alphas[i] < C):
                    b = b1
                elif (alphas[j] > 0) and (alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d; i: %d; paris changed: %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


def cal_W(alphas, dataMatin, classLabels):   #根据alphas、样本X、Y来计算W
    """
    根据alphas、各样本输入dataMatin（100X2）和标签classLabels（100X1）来计算模型原始参数w(2X1)
    """
    X = np.mat(dataMatin)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    W = np.zeros((n, 1))
    for i in range(m):
        W += np.multiply(alphas[i]*labelMat[i], X[i, :].transpose())  # alphas[i]、labelMat[i]、X[i, :]进行矩阵对应位置相乘，然后各样本相加，即W
    return W


if __name__ == "__main__":
    """加载数据集，dataSet（100X2）"""
    dataSet, labels = loadDataset("testSet.txt")
    """计算模型参数b, alphas"""
    b, alphas = smoSimple(dataSet, labels, 0.6, 0.001, 40)
    print('b is :', b)
    print('alphas is :', alphas)
    """计算模型参数W"""
    w = cal_W(alphas, dataSet, labels)
    print('w is ', w)
    """计算训练样本的模型计算值moni"""
    dataMatrix = np.mat(dataSet)
    labelMat = np.mat(labels).transpose()
    moni = dataMatrix * w + b          # 举例比较样本的计算值和实际值
    shiji = labelMat
    moni = np.array(moni)
    shiji = np.array(shiji)
    # print(moni.dtype, shiji.dtype)   # 数组元素类型
    # shiji.astype('float64')          # 改变数组元素类型
    res = np.hstack((moni, shiji))     # 数组横向叠加
    print(res)


"""
数组array与矩阵mat的*，dot()，multiply()，matmul()的区别(元素a与b)
array & '*'：即a与b的对应位置相乘，需要保证a、b同样大小，返回数组的大小同a
mat & '*'： 即矩阵的乘法，需要保证a的列与b的行相同，返回矩阵的行同a，列同b

array & 'dot()'：类似矩阵的dot()，需要保证a的列与b的行相同，只是返回的是数组，且行同a，列同b；不推荐使用
mat & 'dot()'： 即矩阵的乘法,同 mat & '*'

array & 'multiply()'：类似"array & '*'"，即数组a、b的对应位置相乘，需要保证a、b同样大小
mat & 'multiply()'：类似"array & '*'"，即矩阵a、b的对应位置相乘，需要保证a、b同样大小

array & 'matmul()'：类似矩阵的dot()，需要保证a的列与b的行相同，只是返回的是数组，且行同a，列同b；
mat & 'matmul()'：即矩阵的乘法,同 mat & '*'
"""

