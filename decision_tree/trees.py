import numpy as np
import os
import math


def cal_shannonent(dataset):  #计算熵：数据集各输出Y的可能性。。。之和
    """
    熵：各种标签y的熵之和，某个标签y的熵：-p(y)*log(p(y))
    :param dataset: 数据集，包含输入和标签
    :return: 该数据集的熵
    """
    num_ent = len(dataset)
    laber_cnt = {}
    shannonent = 0.0
    for i in range(num_ent):
        cur_label = dataset[i, -1]
        laber_cnt[cur_label] = laber_cnt.setdefault(cur_label, 0) + 1
    for key in laber_cnt:
        """计算各标签的概率及其熵"""
        pro = laber_cnt[key]/num_ent
        shannonent = shannonent - pro * math.log(pro, 2)
    return shannonent


def split_dataset(dataset, fec, value):
    """"
    按照特征划分数据集，输入：数据集，某特征（X的某一列），该特征的值。返回划分后数据集
    返回数据集：删除特征列，删除特征列的值非指定value的行
    """
    ret_dataset = []
    for ent in dataset:
        if ent[fec] == value:
            ret_dataset.append(ent)     #删除特征列的值非指定value的行（ret_dataset是list，列表元素为矩阵）
    ret_dataset = np.delete(ret_dataset, fec, axis=1)   #删除特征列（对list类型删除也行？）
    return ret_dataset


def choose_better_fec(dataset):
    """
    选择最佳特征：输入数据集dataset（含输出Y），返回最佳特征（dataset的列索引）
    1.计算熵baseshannon：各输出Y的可能性。。。之和
    2.计算每个特征的shannonent，并计算熵增益
    2.1计算某特征各取值的sub_dataset、该取值概率pro
    2.2计算pro * cal_shannonent(sub_dataset)，并把各取值的计算结果相加
    3.熵增益排序，选择增益最大的特征
    """
    """计算熵baseshannon"""
    num_fec = len(dataset[0]) - 1    # dataset[0]表示矩阵dataset的第一行，num_fec为特征个数：2
    baseshannon = cal_shannonent(dataset)
    shannonent = 0
    info_gain_list = []
    """计算每个特征的shannonent"""
    for i in range(num_fec):      #对于每一个特征i
        fec_list = [enc[i] for enc in dataset]   #计算特征i下特征值列表
        fec_uniq = set(fec_list)                 #计算特征i下所有特征值
        # print(fec_list)
        # print(fec_uniq)
        for value in fec_uniq:
            sub_dataset = split_dataset(dataset, i, value)
            # print(sub_dataset)
            pro = len(sub_dataset)/float(len(dataset))
            # print(pro)
            """特征i的熵：特征各取值value的熵之和；某value的熵：pro * cal_shannonent(sub_dataset)"""
            shannonent = shannonent + pro * cal_shannonent(sub_dataset) #shannonent：该特征下的熵
            # print(shannonent)
        info_gain = baseshannon - shannonent      #该特征下的熵增益
        info_gain_list.append(info_gain)          #增益列表
    return info_gain_list.index(max(info_gain_list))    #返回info_gain_list最大值对应位置索引


def major_cnt(classlist):   #入参：输出Y一列；返回输出Y次数最多的Y值
    class_cnt = {}
    for i in classlist:
        class_cnt[i] = class_cnt.setdefault(i, 0) + 1
    sort_class_cnt = sorted(class_cnt.items(), key=lambda item: item[1], reverse=True)
    return sort_class_cnt[0][0]


def create_tree(dataset, labels):
    """
    创建决策树：1.选取最佳特征，根据该特征取值数目分叉；2.对各个分叉求子矩阵，迭代第1、2步骤
    dataset：输入数据集，包含Y
    labels：各特征X的代表的真实意义，特征名称列表
    """
    classlist = [ent[-1] for ent in dataset]   #输出Y的列
    if classlist.count(classlist[0]) == len(classlist):   #如果输出Y只有一种值
        return classlist[0]
    if len(dataset[0]) == 1:                              #如果所有特征都筛选完毕，只剩下Y
        return major_cnt(classlist)
    best_fec = choose_better_fec(dataset)
    best_fec_label = labels[best_fec]
    mytree = {best_fec_label: {}}
    del(labels[best_fec])
    fec_list = [enc[best_fec] for enc in dataset]  # 计算best_fec下特征值列表
    fec_uniq = set(fec_list)  # 计算best_fec下所有特征值
    for value in fec_uniq:
        sublabels = labels[:]
        mytree[best_fec_label][value] = create_tree(split_dataset(dataset, best_fec, value), sublabels)
    return mytree


if __name__=='__main__':
    labels = ['no surfacing', 'flippers']
    dataset = np.array([1,1,1,1,1,1,1,0,0,0,1,0,0,1,0]).reshape(5, 3)
    print(dataset)
    print(create_tree(dataset, labels))



