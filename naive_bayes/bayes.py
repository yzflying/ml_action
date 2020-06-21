import numpy as np
import math


def loadDateSet():   #创建实验样本，postingList是拆分后的文档的词条集合，classVec是类别标签
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocablist(dataSet):  #dataSet去重，返回dataset中所有不重合词条
    vocabSet = set([])
    for documnet in dataSet:
        vocabSet = vocabSet | set(documnet)  #利用set去重，取并集
    return list(vocabSet)


def setOfWords2Vec(vocablist, inputSet):
    """
    vocablist：已知敏感词汇表
    inputSet：待鉴别拆分后的文档的词条
    returnVec：0、1向量，表示vocablist中的敏感词是否存在
    """
    returnVec = [0]*len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
        else:
            print("The Word:%s is not in my vocabulary" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    trainMatrix：输入处理后的训练对象集，每个对象长度为myVocablist，元素为0或1
    trainCategory：输入训练对象的分类，0或1
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    """p_abusive:分类为1的样本占所有样本的比例"""
    p_abusive = sum(trainCategory)/float(numTrainDocs)
    p_0Num = np.zeros(numWords)
    p_1Num = np.zeros(numWords)
    p_0Denom = 0.0
    p_1Denom = 0.0
    for i in range(numTrainDocs):
        """对每个训练样本，如果标签为1：trainMatrix行的对应元素相加（行向量，长度为myVocablist，元素值介于：0-分类为1的对象个数）"""
        if trainCategory[i] == 1:
            p_1Num += trainMatrix[i]
            p_1Denom += sum(trainMatrix[i])  #分类为1的所有对象在myVocablist词汇中的总数，即p_1Num各元素相加之和
        else:
            p_0Num += trainMatrix[i]
            p_0Denom += sum(trainMatrix[i])
    print('p_1Num is :', p_1Num)
    """p_1Vect：（行向量，长度为myVocablist，元素值介于：0-1的小数）表示所有标签为1的样本，各词汇出现概率"""
    p_1Vect = p_1Num/p_1Denom
    p_0Vect = p_0Num/p_0Denom
    return p_0Vect, p_1Vect, p_abusive


def classifyNB(vec2Classify, p_0Vect, p_1Vect, p_abusive):
    p1 = sum(vec2Classify*p_1Vect) + math.log(p_abusive)
    p0 = sum(vec2Classify*p_0Vect) + math.log(p_abusive)
    if p1 > p0:
        return p1
    else:
        return p0


if __name__ == "__main__":
    """加载数据集。listOPosts为6Xn二维数组，即6个文档doc及其单词，元素为单词, listClasses为len==6的列表，代表对于文档的分类，用0、1表示"""
    listOPosts, listClasses = loadDateSet()
    """myVocablist是一个去重后的单词列表，涵盖了所有文档单词"""
    myVocablist = createVocablist(listOPosts)
    print("myVocablist is :", myVocablist)
    """对listOPosts中的每个文档，判断每个单词是否在myVocablist出现，标1；生成trainMat矩阵6Xlen(myVocablist)"""
    trainMat = []      # 创建一个空矩阵，行数为对象个数，每行长度为myVocablist，存储myVocablist中的词是否该对象出现
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocablist, postinDoc))
    print("trainMat is :", np.array(trainMat))
    """根据训练输入矩阵trainMat 6Xlen(myVocablist)及其标签"""
    p_0Vect, p_1Vect, p_abusive = trainNB0(trainMat, listClasses)
    print("p_1Vect is :", p_1Vect)
    print("p_0Vect is :", p_0Vect)
    print("p_abusive is :", p_abusive)





