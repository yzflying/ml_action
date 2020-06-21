import numpy as np


def loadDataSet(fileName):
    fr = open(fileName)
    stringArr = [line.strip().split('\t') for line in fr.readlines()]    #str.strip()移除字符串str的首位字符(默认是空格)，此处\t相当于tab
    m = len(stringArr)        # 获取样本个数
    dat_list = [float(val) for line in stringArr for val in line]   # 将每个样本各元素转为float格式,生成结果为1维列表
    dat_arr = np.array(dat_list).reshape(m, -1)         #生成array，shape为80 X 2
    return np.mat(dat_arr)    # 返回数据集的矩阵形式


def distEclud(vecA, vecB):
    return np.sum(np.power(vecA - vecB, 2))  # la.norm(vecA-vecB),计算俩向量的距离


def randCent(dataSet, k):
    """返回k个随机样本,随机样本的属性值在数据集样本属性值内"""
    n = np.shape(dataSet)[1]    # 数据集为2列
    centroids = np.mat(np.zeros((k, n))) #创建K行2列的零矩阵
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))   # np.random.rand(k, 1)返回一个K*1的array，元素服从“0~1”均匀分布
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    随机生成k个簇心，进行更新
    :param dataSet: 样本
    :param k: 簇心个数
    :param distMeas: 计算簇心距离的函数
    :param createCent: 创建初始簇心的函数
    :return: centroids, clusterAssment，k个簇心和m个样本的簇信息
    """
    m = np.shape(dataSet)[0]
    """创建一个m*2矩阵，第一列保存样本所属簇，第二列保存样本到簇心的距离平方"""
    clusterAssment = np.mat(np.zeros((m,2)))
    """生成随机k个样本"""
    centroids = createCent(dataSet, k)
    """定义clusterChanged，遍历m个样本，只要任一改变了所属簇心，则遍历完1轮后，再更新簇心；重新下一次遍历，直到所有样本不改变簇心"""
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        """遍历m个样本"""
        for i in range(m):
            minDist = float("inf")
            minIndex = -1
            """遍历k个簇心，找到最近的"""
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            """如果簇心改变了，clusterChanged改为True，簇信息存入clusterAssment"""
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        """遍历完整1轮后，更新簇心情况，再根据clusterChanged决定是否遍历下一轮"""
        for cent in range(k):
            """返回簇心索引为当前簇cent的样本，对应索引"""
            dataSet_cent = np.nonzero(clusterAssment[:, 0].A == cent)
            ptsInClust = dataSet[dataSet_cent[0]]  # get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis=0) # assign centroid to mean
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    二分法：将所有点看成一个簇，当簇小于k时，
    计算每一个簇的误差平方和ssh，选择ssh最大的簇进行再分，直到簇数达到k
    ps:刚开始可以取较大的k，计算簇数增大的时候，总的ssh是否依次减小，当不再明显减小时，k合适
    :param dataSet: 样本
    :param k: 簇心个数
    :param distMeas: 计算簇心距离的函数
    :return: k个簇心和m个样本的簇信息
    """
    m = np.shape(dataSet)[0]
    """创建一个m*2矩阵，第一列保存样本所属簇，第二列保存样本到簇心的距离平方"""
    clusterAssment = np.mat(np.zeros((m, 2)))
    """初始簇心，即k=1时簇心"""
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]  # 将1*2的矩阵转换为list后，取[0]j即矩阵第1行
    centList =[centroid0]
    """计算m个样本与簇心的距离，并保存在clusterAssment"""
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
    """当簇心个数小于K,循环"""
    while (len(centList) < k):
        lowestSSE = float("inf")
        for i in range(len(centList)):
            """step1：对列表中的每个簇心i，对簇i内样本尝试一分为二，计算ssh"""
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A==i)[0], :]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum   簇i被划分后的俩子簇ssh之和
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A!=i)[0],1])  # 此次未被划分的簇（即非簇i）的ssh之和
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            """如果总的ssh有改善"""
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i                  # 被划分的簇i
                bestNewCents = centroidMat           #簇i分开后的俩子簇心
                bestClustAss = splitClustAss.copy()  #原属簇i的样本的在新子簇下的归属与距离信息
                lowestSSE = sseSplit + sseNotSplit   # 总ssh
        """step2：选择好了被划分的簇后...更新簇心和列表centList"""
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return np.mat(centList), clusterAssment


if __name__ == "__main__":
    k = 4     # 定义簇心个数
    """导入数据集80*2"""
    dataMat = loadDataSet("testSet.txt")
    """计算样本dataMat的簇情况,centroids是k个簇心；clusterAssment是m*2矩阵，保存样本所属簇与距离"""
    centroids, clusterAssment = kMeans(dataMat, k, distMeas=distEclud, createCent=randCent)
    """对比二分法下的簇情况"""
    biKmeans(dataMat, k, distMeas=distEclud)
