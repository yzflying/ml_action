import numpy as np
import math
import random


def loadSimpData():
    datMat = np.matrix([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]
    ])
    classLables = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLables


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    输入：样本dataMatrix，特征列dimen，阈值threshVal，方向threshIneq（lt或gt）
    功能：对于dataMatrix的dimen列各元素，如果不大于threshVal则标记-1，否则标记1，保存在retArray
    输出：retArray（mX1），元素为1或-1
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0    #选择矩阵dataMatrix的dimen列元素<=threshVal的那些行，置-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    计算特定样本权重下的，最佳决策树bestStump
    输入：样本dataArr，样本对应分类classLabels，权重向量D
    功能：改变每个特征列i、阈值threshVal、方向threshIneq来构造决策树，选择最佳决策树
    输出：最佳决策树bestStump，及其决策结果bestClasEst和误差minError
    """
    """训练集的输入数据dataMatrix、标签labelMat"""
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    """定义一些超参数"""
    numSteps = 10.0     # 步数，即每个特征选取的阈值个数；根据该值、特征取值范围可以计算步长
    bestStump = {}      # 空字典，保存决策树三要素的值
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float("inf")   # 保存最小加权错误率(最大正确率)
    """step1：对每一个属性遍历"""
    for i in range(n):
        """step2：计算该属性取值范围，步长"""
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps    #对每个特征列i，根据元素范围、步数计算步长
        """step3：遍历该属性下的每一个分割点"""
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                """step4：计算分割点值threshVal，对应分类器的样本预测predictedVals"""
                threshVal = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) #对i, threshVal, inequal下的决策结果（mX1），元素1或-1
                """step5：定义mX1矩阵，元素为0或1，保存该分类器的预测情况"""
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0    #对errArr矩阵决策正确的样本行置0
                """step6：计算该分类器的加权正确率weightedError"""
                weightedError = D.transpose() * errArr
                if weightedError < minError:
                    """step7：更新值，最小加权错误率minError、bestClasEst决策结果、决策树bestStump"""
                    minError = weightedError
                    bestClasEst = predictedVals.copy()  # 更新最佳决策树的决策结果（mX1），元素1或-1
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['inequal'] = inequal      # 更新最佳决策树的三要素i, threshVal, inequal
    return bestStump, minError, bestClasEst


def sign(cal_Label):   #对计算得到的label归类，小于0置-1，大于0置1
    retArray = np.ones((np.shape(cal_Label)[0], 1))
    retArray[cal_Label[:, 0] <= 0] = -1.0
    return retArray


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    输入：样本dataArr，样本对应分类classLabels，迭代次数numIt
    功能：更新并迭代权重向量D，依据buildStump函数计算最佳决策树bestStump信息及其权重alpha
    输出：weakClassArr，保存各个最佳决策树的bestStump，及其权重alpha
    """
    """定义空字典，保存各基分类器bestStump的信息"""
    weakClassArr = []
    """初始化各样本权重D（mX1）"""
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)     #定义初始权重向量，各样本权重为1/m
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        """step1：计算基分类器bestStump及其加权错误率error、决策结果classEst（mX1），元素1或-1"""
        bestStump, error, classEst = buildStump(dataArr, classLabels, D) #迭代D，计算bestStump, error, classEst
        # print('D is:', D)
        # print('error is:', error)
        """step2：依据error计算该基分类器系数alpha"""
        alpha = float(0.5*math.log((1.0-error)/max(error, 1e-16)))
        # print('alpha is:', alpha)
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)    #将决策树bestStump信息添加到weakClassArr
        # print('classEst is:', classEst)
        """step3：更新样本权重D，expon为（mX1）数组"""
        expon = np.multiply(-1*alpha*np.mat(classLabels).transpose(), classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()                     #更新权重向量D（D与alpha的关系？？？）
        """step4：汇总基分类器，得到模型的输出值sign(aggClassEst)，计算模型错误率errorRate"""
        aggClassEst += alpha*classEst     #最终分类结果+=当前决策树系数*当前决策树分类结果
        # print('aggClassEst is:', aggClassEst)
        aggErrors = np.multiply(sign(aggClassEst) != np.mat(classLabels).transpose(), np.ones((m, 1)))  #计算决策错误个数
        errorRate = aggErrors.sum()/m
        # print('errorRate is:', errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(data_test, weakClassArr):
    """
    输入：待测试样本data_test，各最佳决策树weakClassArr
    功能：（测试函数）利用stumpClassify对各最佳决策树参数决策，得到分类结果，按权重累加
    输出：每个待测试样本data_test的测试结果sign(aggClassEst)
    """
    dataMatrix = np.mat(data_test)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))    #保存每个测试样本的计算值（初始化为0）
    for i in range(len(weakClassArr)):
        classEst = stumpClassify(dataMatrix, weakClassArr[i]['dim'], weakClassArr[i]['thresh'], weakClassArr[i]['inequal'])   #依据决策树参数的决策结果
        aggClassEst += weakClassArr[i]['alpha'] * classEst         #样本最终决策值：累加当前决策树权重*当前决策树分类结果
        print(classEst)
        print(aggClassEst)
    return sign(aggClassEst)        #测试样本计算值的+1、-1转化


if __name__ == "__main__":
    dataSet, labels = loadSimpData()
    print(dataSet, labels)
    weakClassArr = adaBoostTrainDS(dataSet, labels, numIt=9)
    print(weakClassArr)
    res = adaClassify([[0, 0], [5, 5]], weakClassArr)
    print(res)




