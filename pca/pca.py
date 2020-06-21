import numpy as np
import math


def loadDataSet(fileName):
    fr = open(fileName)
    stringArr = [line.strip().split(' ') for line in fr.readlines()]    #str.strip()移除字符串str的首位字符(默认是空格)
    m = len(stringArr)        # 获取样本个数
    dat_list = [float(val) for line in stringArr for val in line]   # 将每个样本各元素转为float格式,生成结果为1维列表
    dat_arr = np.array(dat_list).reshape(m, -1)         #生成array，shape为1567 X 590
    return np.mat(dat_arr)    # 返回数据集的矩阵形式


def pca(dataMat, topNfeat=6):
    """计算各属性平均值，meanVals是1 X 590的mat"""
    meanVals = np.mean(dataMat, axis=0)
    """中心化：对各样本，减去各属性平均值，得到meanRemoved 1567 X 590"""
    meanRemoved = dataMat - meanVals
    """
    通过正交变换，将590维数据(X=m*590)降至6维(Y=m*6)，需要找到6条向量(转换矩阵W)，使得X在W上的投影方差 (W(X.T))*(X(W.T))最大；
    W即由协方差矩阵(X.T)*X 的6个最大特征值对应的特征向量组成，(W=6*590)，输出矩阵Y=X*W.T
    """
    # 求协方差矩阵(X.T)*X，即590*590；此处rowvar为0表示X的每行为1个样本；默认rowvar=1表示每列为1个样本
    covMat = np.cov(meanRemoved, rowvar=0)
    # 求矩阵的特征值、特征向量;eigVals为array类型；eigVects为mat类型，每列表示一个特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    """通过特征值可以看出该特征对应方差大小，前6个方差大小达到总方差大小96.8%"""
    about = sum(eigVals[:6]) / sum(eigVals)
    print(about)
    eigValInd = np.argsort(eigVals)           # 特征值从小到大排序后的index
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 倒序取topNfeat个索引
    redEigVects = eigVects[:, eigValInd]       # 取topNfeat个索引的对应特征向量(590*6)
    lowDDataMat = meanRemoved * redEigVects   # 转换后的矩阵Y 1567*6
    return lowDDataMat


def replaceNanWithMean():
    """加载数据集secom.data，1567 X 590，并对缺失值填充处理"""
    datMat = loadDataSet("secom.data")        # datMat矩阵 1567 X 590
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        # 判断属性i的各样本值是否NAN，返回True或False矩阵（1X1567）（np.isnan的入参为list、array或mat，但要为1行）
        dat_i = np.isnan(datMat[:, i].T)
        # 对矩阵True或False取反，即空值为False
        dat_i = ~dat_i
        # 返回True(即非NAN)值的索引，返回值类型为元组，包含俩列表，对应行列索引(array([0, 0, 0, 0, 0, 0], dtype=int64), array([ 21,  66, 117, 538, 725, 885], dtype=int64))
        non_zero_inx = np.nonzero(dat_i)
        # 取出属性i的非空值样本行，求平均值
        meanVal = np.mean(datMat[non_zero_inx[1], i])  # np.nonzero:返回数组中非0元素的索引
        # 返回空值索引zero_inx，填充meanVal
        zero_inx = np.nonzero(~dat_i)
        datMat[zero_inx[1], i] = meanVal  #set NaN values to mean
    return datMat


if __name__ =="__main__":
    """导入数据文件，并对空值填充该属性平均值处理"""
    datMat = replaceNanWithMean()
    datMat_new = pca(datMat)
    print(datMat_new)
