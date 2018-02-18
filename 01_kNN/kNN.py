from numpy import *
import operator


def createDataSet():
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


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


if __name__ == "__main__":
    group, lables = createDataSet()
    while True:
        x = float(input("x:"))
        y = float(input("y:"))
        res = classify0([x, y], group, lables, 3)
        print("res:", res)
        print("**********************************")
