import numpy as np
import math
import re
import os
import random


def loadDateSet():   #创建实验样本，postingList是拆分后的文档的词条集合，classVec是类别标签
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocablist(dataSet):  #dataSet去重，返回dataset中所有不重合词条(此处dataSet可以为嵌套列表)
    vocabSet = set([])
    for documnet in dataSet:
        vocabSet = vocabSet | set(documnet)  #利用set去重，取并集
    return list(vocabSet)


def bagOfWords2Vec(vocablist, inputSet):
    """
    vocablist：已知敏感词汇表
    inputSet：待鉴别拆分后的文档的词条
    returnVec：0、1向量，表示vocablist中的敏感词是否存在,存在次数
    """
    returnVec = [0]*len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    trainMatrix：输入处理后的训练对象集，每个对象长度为myVocablist，元素为0或1
    trainCategory：输入训练对象的分类，0或1
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    p_abusive = sum(trainCategory)/float(numTrainDocs)   #分类为1的对象占所有对象的比例
    p_0Num = np.ones(numWords)
    p_1Num = np.ones(numWords)
    """
    将p_0Num初始化为全1，p_0Denom初始化为2.0，避免某个参数为0造成累乘的最后结果为0（拉普拉斯平滑）
    p_0Denom初始化值应该是1.0*len(trainMatrix)  ？？？
    """
    p_0Denom = 2.0
    p_1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p_1Num += trainMatrix[i]       #分类为1的对象，trainMatrix行的对应元素相加（行向量，长度为myVocablist，元素值介于：0-分类为1的对象个数）
            p_1Denom += sum(trainMatrix[i])  #分类为1的所有对象在myVocablist词汇中的总数，即p_1Num各元素相加之和
        else:
            p_0Num += trainMatrix[i]
            p_0Denom += sum(trainMatrix[i])
    # print(p_1Num)
    p_1Vect = np.log(p_1Num/p_1Denom)                #表示分类为1的所有对象，myVocablist中各元素占比(np.log)
    p_0Vect = np.log(p_0Num/p_0Denom)
    """"
    取log，避免数据太小下溢
    """
    return p_0Vect, p_1Vect, p_abusive


def classifyNB(vec2Classify, p_0Vect, p_1Vect, p_abusive):  #判断向量vec2Classify的概率
    """
    :param vec2Classify: 待预测样本，1Xlen(myVocablist)，元素为0或1
    :param p_0Vect:
    :param p_1Vect:
    :param p_abusive:
    :return:
    样本每个单词乘以该单词在0、1分类中出现的概率，求和
    """
    p1 = sum(vec2Classify*p_1Vect) + math.log(p_abusive)
    p0 = sum(vec2Classify*p_0Vect) + math.log(1.0 - p_abusive)
    if p1 > p0:
        return 1
    else:
        return 0


def textParse(text):        #根据句子(每个邮件)生成词汇列表
    re_split = re.compile('[^a-zA-Z0-9]')      #非词汇、数字的字符
    a = re_split.split(text)
    b = [word.lower() for word in a if len(word) > 0]
    return b


def filepath_to_Mat(filepath):             #文件夹转换为二维列表，每个文件对应一行
    filelist = os.listdir(filepath)
    docList = []
    for file in filelist:
        concept = open("%s/%s" % (filepath, file)).read()   #打开文件-另存为-编码格式，可以查看编码格式
        docList.append(textParse(concept))
    return docList


if __name__ == "__main__":
    """创建两种类型的邮件样本矩阵docList_ham、docList_spam，元素为单词"""
    filepath_ham = "email\ham"
    filepath_spam = "email\spam"
    docList_ham = filepath_to_Mat(filepath_ham)       #ham类型各邮件词汇列表
    docList_spam = filepath_to_Mat(filepath_spam)      #spam类型各邮件词汇列表
    vocabList = createVocablist(docList_ham + docList_spam)   #词汇集
    """对docList中的每个文档，判断每个单词是否在vocabList出现，标1；生成trainSet矩阵mXlen(myVocablist)"""
    trainSet = []          #用于保存所有50个邮件词汇列表
    classList = []         #用于保存所有50个邮件的分类
    for wordList in docList_ham:
        trainSet.append(bagOfWords2Vec(vocabList, wordList))
        classList.append(0)
    for wordList in docList_spam:
        trainSet.append(bagOfWords2Vec(vocabList, wordList))
        classList.append(1)
    """对trainSet、classList随机抽取10个样本标签作为测试集"""
    textSet = []
    text_classList = []
    for i in range(10):
        randIndex = random.randint(0, len(trainSet)-1)
        # print(randIndex)
        textSet.append(trainSet[randIndex])
        text_classList.append(classList[randIndex])
        del (trainSet[randIndex])
        del (classList[randIndex])

    p_0Vect, p_1Vect, p_abusive = trainNB0(np.array(trainSet), np.array(classList))
    cnt = 0
    for i in range(10):
        if text_classList[i] == classifyNB(textSet[i], p_0Vect, p_1Vect, p_abusive):
            cnt += 1
    print("10 个样本预测正确个数为：", cnt)






