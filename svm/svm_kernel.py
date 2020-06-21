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


def selectJrand(i, m):      #随机选择i和j
    j = i
    while (j == i):
        j = random.randint(0, m-1)
    return j


def clipAlpha(a, H, L):  #限值
    if a > H:
        a = H
    if a < L:
        a = L
    return a


def smoSimple(dataMatin, classLabels, C, toler, maxIter): #数据集；输出类别；C；容错率；最大循环次数
    dataMatrix = np.mat(dataMatin)      #m*n
    labelMat = np.mat(classLabels).transpose()  #m*1
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))   #m*1,待优化的alphas向量
    iter = 0
    eCache = np.mat(np.zeros((m, 2)))  # m*2,用于保存各个alphaJ对应的误差Ej，及其标志位


    def selectJ(i):     #新增函数，用于找出步长最大的j
        maxDeltaE=0
        Ej=0
        maxK = -1
        # fXi = float((np.multiply(alphas, labelMat).transpose()) * dataMatrix * (dataMatrix[i, :].transpose())) + b
        fXi = float((np.multiply(alphas, labelMat).transpose()) * K[:, i]) + b
        Ei = fXi - float(labelMat[i])
        eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(eCache[:,0].A)[0]
        if len(validEcacheList)>1:
            for k in validEcacheList:
                if k == i:
                    continue
                # fXk = float((np.multiply(alphas, labelMat).transpose()) * dataMatrix * (dataMatrix[k, :].transpose())) + b
                fXk = float((np.multiply(alphas, labelMat).transpose()) * K[:, k]) + b
                Ek = fXk - float(labelMat[k])
                deltaE = abs(Ei-Ek)
                if deltaE>maxDeltaE:
                    maxDeltaE=deltaE
                    Ej = Ek
                    maxK = k
            return maxK, Ej
        else:
            j = selectJrand(i, m)
            # fXj = float((np.multiply(alphas, labelMat).transpose()) * dataMatrix * (dataMatrix[j, :].transpose())) + b
            fXj = float((np.multiply(alphas, labelMat).transpose()) * K[:, j]) + b
            Ej = fXj - float(labelMat[j])
            return j, Ej


    def Cal_K(kTup):
        """rbf核函数"""
        X = np.mat(dataMatin)
        K = np.mat(np.zeros((m, m)))
        for i in range(m):
            for j in range(m):
                deltaRow = X[j, :] - X[i, :]
                K[i, j] = math.exp((deltaRow*(deltaRow.transpose()))/(-1*kTup**2))
        return K
    K = Cal_K(1.3)         # 先行运算得到核函数K，kezei参数暂取1.3

    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # fXi = float((np.multiply(alphas, labelMat).transpose()) * dataMatrix * (dataMatrix[i, :].transpose())) + b
            fXi = float((np.multiply(alphas, labelMat).transpose()) * K[:, i]) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):   #对alphas[i]对应实例，如果Ei误差较大且alphas[i]有优化余地
                j,Ej = selectJ(i)
                # j = selectJrand(i, m)    #随机选取J
                # fXj = float((np.multiply(alphas, labelMat).transpose()) * dataMatrix * (dataMatrix[j, :].transpose())) + b
                # Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[i] - alphas[j])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    print('L==H')
                    continue
                # eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].transpose() - dataMatrix[i, :] * dataMatrix[i, :].transpose() - dataMatrix[j, :] * dataMatrix[j, :].transpose()
                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei-Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold-alphas[j])
                # b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].transpose() - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].transpose()
                # b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].transpose() - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].transpose()
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * K[i, i] - labelMat[j] * (alphas[j] - alphaJold) * K[i, j]
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * K[i, j] - labelMat[j] * (alphas[j] - alphaJold) * K[j, j]
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
    X = np.mat(dataMatin)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    W = np.zeros((n, 1))
    for i in range(m):
        W += np.multiply(alphas[i]*labelMat[i], X[i, :].transpose())
    return W


if __name__ == "__main__":
    dataSet, labels = loadDataset("testSet.txt")
    b, alphas = smoSimple(dataSet, labels, 0.6, 0.001, 40)
    print("***************")
    print(b)
    print("***************")
    print(alphas)

    w = cal_W(alphas, dataSet, labels)
    print("***************")
    print(w)

    dataMatrix = np.mat(dataSet)
    labelMat = np.mat(labels).transpose()
    moni0 = dataMatrix[0] * w + b          #举例比较0样本的计算值和实际值
    shiji0 = labelMat[0]
    print(moni0, shiji0)


