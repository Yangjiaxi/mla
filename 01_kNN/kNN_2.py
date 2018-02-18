from numpy import *
import operator
import matplotlib.pyplot as plt


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape返回这个array的形状(4, 2)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # tile将inX沿着行重复dataSetSize次
    # 再与dataSet中各个元素逐个相减
    sqDiffMat = diffMat ** 2
    # 将每个元素计算平方
    sqDistances = sqDiffMat.sum(axis=1)
    # 沿着行求和
    distances = sqDistances ** 0.5
    # 开根号，求出欧氏距离
    sortedDistIndicies = distances.argsort()
    # 得到数组distances元素从大到小排序的索引
    classCount = {}
    # 创建字典
    # 这个for循环统计标签出现的次数
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
        # dict.get(key, default=) 获得字典中key的value，若不存在此key，返回0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # operator.itemgetter(1)表示获得对象域1的内容
    # list(map(lambda ex: operator.itemgetter(1)(ex), classCount.items())) -> [2, 1]
    return sortedClassCount[0][0]


def file2mat(filename):
    # 文件有很多行
    # 每行四个数据
    # 最后一个数据表示对这个人的感觉
    # 前三个float分别表示
    #   每年获得的飞行常客里程数
    #   玩视频游戏所耗时间百分比
    #   每周消费的冰激凌公升数
    fr = open(filename)
    arrayOLines = fr.readlines()  # 以行的方式读取全部
    numberOfLines = len(arrayOLines)  # 获得行数
    returnMat = zeros((numberOfLines, 3))  # 用0填充行数*3的矩阵
    classLabelVector = []  # 标签向量
    index = 0  # 自增索引
    for line in arrayOLines:
        line = line.strip()  # 去掉前后的空白字符
        listFromLine = line.split('\t')  # 以tab划分数据
        # 形成一个有四个元素的list
        returnMat[index, :] = listFromLine[0:3]
        # 使用listFromLine[0,1,2]四个元素填充returnMat的第index行
        classLabelVector.append(int(listFromLine[-1]))
        # 把listFromLine的最后一个元素接到标签向量之后
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):  # 特征值归一化，平均权值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    # 构建一个dataSet同型的零矩阵
    m = dataSet.shape[0]
    # 得到行数
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 构造一个m行1列的矩阵，值全为min，再用dataSet对应相减
    normDataSet = normDataSet / tile(ranges, (m, 1))
    # 同理构造ranges，相除
    # 此时每个元素e都是(e-minVals)/ranges
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    # 测试率，1-hoRatio表示用于训练的数据占比，hoRatio比例的数据将用于测试正确率
    datingDataMat, datingLabels = file2mat("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 获得数据样本数
    numTestVecs = int(m * hoRatio)
    # 前hoRatio个样本用于测试，[hoRatio+1:m]的数据用于训练，所以numTestVecs保存了用于测试的数据数量
    errorCount = 0.0
    # 统计错误格数，使用浮点型避免之后的变量提升操作
    for i in range(numTestVecs):
        print("%2d" % i, end="# ")
        classifierRes = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # 使用第i个数据进行测试，测试对象是使用[numTestVecs:m]的数据生成的结果
        print("Came back with %d, real is %d" % (classifierRes, datingLabels[i]), end=" ")
        if (classifierRes != datingLabels[i]):  # 统计错误
            errorCount += 1.0
            print((datingDataMat[i][0], datingDataMat[i][1]));
        else:
            print()
    print("Error Rate: %f" % (errorCount / float(numTestVecs)))


if __name__ == "__main__":
    datingClassTest()
    # mat, lab = file2mat("datingTestSet2.txt")
    # fig = plt.figure();
    # ax = fig.add_subplot(111)
    # ax.set_title("Larger, the Charmer")
    # lab = array(lab)
    # idx_1 = where(lab == 1)
    # p1 = ax.scatter(mat[idx_1, 0], mat[idx_1, 1], marker='*', color='r', label='1', s=20)
    # idx_2 = where(lab == 2)
    # p2 = ax.scatter(mat[idx_2, 0], mat[idx_2, 1], marker='o', color='b', label='2', s=10)
    # idx_3 = where(lab == 3)
    # p3 = ax.scatter(mat[idx_3, 0], mat[idx_3, 1], marker='+', color='g', label='3', s=30)
    # plt.legend(loc='upper right')
    # plt.xlabel("Flight (miles)")
    # plt.ylabel("Video Game(%)")
    # plt.show()
