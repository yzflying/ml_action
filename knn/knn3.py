import numpy as np
import os


def classify0(inX, dataSet, labels, k):
    """
    KNN邻近算法，分类器
    inX:输入待测试n维数据（1Xn）； dataSet：输入学习n维数据集； labels：输出分类标签； k：选取距inX邻近的k个学习数据集；
    本函数计算inX的的输出分类标签
    """
    """计算与数据集的距离并排序"""
    dataSetsize = dataSet.shape[0]    #计算数据集的行数
    inXX = np.tile(inX, (dataSetsize, 1))   #对inX转化为（4*2）的array          注意inX*dataSetsize中只有inX是list才正确，inX是矩阵时，*dataSetsize不会改变inX的shape
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


def filepath_to_Mat(filepath):             #文件夹转换为矩阵
    """
    :param filepath: 数据集文件夹名称
    :return:trainingMat：样本标签集(mX1025)，np.array类型
    """
    filelist = os.listdir(filepath)       # 列出filepath文件夹下的所有文件名称列表
    trainingMat = np.zeros((len(filelist), 1025))   # 每个文件（手写数字图片）由32X32个0或1组成，因此每个图片文件的数据集部分展平为1X1024
    index = 0
    for filename in filelist:
        filename_class = filename.split('_')[0]    # 当前文件对应标签
        trainingMat[index, -1] = filename_class
        file = open("%s/%s" % (filepath, filename))
        for i in range(32):
            fileline = file.readline()
            for j in range(32):
                trainingMat[index, 32 * i + j] = int(fileline[j])    # 读取32X32的图片数据为1X1024
        index = index + 1
    return trainingMat


import time
import multiprocessing


def cal(part_testMat, trainingMat_in, trainingMat_out, k):
    """
    对classify0函数进一步封装，将其判断结果与标签真实值进行比较，返回0或1（在多进程方式中作为任务函数被调用）
    :param e_test: 待测试样本（包含输入和输出标签）
    :param trainingMat_in: 训练集输入
    :param trainingMat_out: 训练集输出
    :param k:
    :return: 0或1
    """
    cnnt = 0
    for i in part_testMat:
        test_out = classify0(i[0:1024], trainingMat_in, trainingMat_out, k)
        if test_out == i[1024]:
            cnnt += 1
    return cnnt


def cal_ok_num(s):
    global ok_cnt
    ok_cnt += int(s)


if __name__ == "__main__":
    for pool_num in range(1, 17):
        for task_num in range(32, 33):
            start_time = time.time()
            k = 3
            """获取训练数据集"""
            trainingMat = filepath_to_Mat("trainingDigits")
            trainingMat_in = trainingMat[:, 0:1024]
            trainingMat_out = trainingMat[:, 1024]
            """获取测试数据集"""
            testMat = filepath_to_Mat("testDigits")
            testMat_in = testMat[:, 0:1024]
            testMat_out = testMat[:, 1024]

            """常规计算，共耗时16 S左右"""
            # test_num = testMat_in.shape[0]   # 测试集样本数量
            # ok_cnt = 0
            # for i in range(test_num):
            #     test_out = classify0(testMat_in[i], trainingMat_in, trainingMat_out, k)
            #     if test_out == testMat_out[i]:
            #         ok_cnt = ok_cnt + 1
            # print("测试集样本数量：", test_num)
            # print("测试结果OK的样本数量：", ok_cnt)

            """利用多核CPU开启多进程，耗时最少达9 S"""
            test_num = testMat_in.shape[0]  # 测试集样本数量
            ok_cnt = 0

            po = multiprocessing.Pool(pool_num)     #创建一个容量为8的进程池po
            interval = test_num // task_num   # 取整，用//
            for i in range(task_num):
                po.apply_async(cal, args=(testMat[i*interval:(i+1)*interval, :], trainingMat_in, trainingMat_out, k,), callback=cal_ok_num)   # 往进程池添加进程或候补进程,callback回调函数，入参为cal函数的输出
            po.close()                       #关闭进程池，不在接收新的候补进程添加
            po.join()                        #主进程运行到此处会阻塞，等待进程池po运行完毕才继续执行

            print("测试集样本数量：", test_num)
            print("测试结果OK的样本数量：", ok_cnt)
            """利用多核CPU开启多进程代码段结束"""

            end_time = time.time()
            print("计算946个测试样本", pool_num, task_num,  "一共耗时(S)：", end_time-start_time)

            """将用时写入文件"""
            list_res = [pool_num, task_num, end_time-start_time]
            pr_res = '---'.join([str(i) for i in list_res])
            with open('compar.txt', mode='a+', encoding="utf-8") as w:
                w.write(pr_res + "\n")
            print("**************************************************************************")

"""
统计：
常量：cpu核数：8；
变量：进程池进程数，任务数：
1.进程池进程数不同时，等待任务数为1时，耗时差别不大，且与常规计算差不多
2.同一进程池数时，等待任务数大于进程池数，且等待任务越多，耗时越长  
3.等待任务数32相同且大于进程池数时，进程池进程数为1时，时间与常规计算差不多，具体进程池数与耗时对应见下：
可以发现：
a>：进程池数2-16，耗时相差不大；b>：进程池数与cpu核数相差较大时，耗时较长；
c>：task进程用不满一个核，那么再多开进程也不能加速；如果主负荷占满一个核，进程开越多切换进程的消耗越大
进程池个数耗时排序：12 14 7 8 10 13 11 16 15 9 4 6 5 3 2 1
1---32---18.357637882232666
2---32---12.632875919342041
3---32---11.224148988723755
4---32---10.918316125869751
5---32---11.030643463134766
6---32---10.81043815612793
7---32---10.504648208618164
8---32---10.48192548751831
9---32---10.54021692276001
10---32---10.576205730438232
11---32---10.452946662902832
12---32---10.289157629013062
13---32---10.560551404953003
14---32---10.324432849884033
15---32---10.457327127456665
16---32---10.657023668289185
"""
