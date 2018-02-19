# 通过给定数据建立决策树
'''
对于当前层数据：
    1.选择使信息增益最大的特征
    2.对这个特征中的每个数据，使用同样的(->1.)算法建立子树
'''

from math import log
import operator


# 计算给定数据集的香农熵
# 香农熵反应数据的混乱程度
# 熵越高，混合的数据越多
# 熵为非负数
# 熵=0代表数据集中仅有一种标签

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 得到向量数据集的大小
    labelCounts = {}  # 字典
    for featVec in dataSet:  # 遍历数据集中每一个向量
        currentLabel = featVec[-1]  # 默认输入数据向量的最后一个维度是标签
        if currentLabel not in labelCounts.keys():  # 如果当前标签还没有记录过
            labelCounts[currentLabel] = 0  # 插入字典
        labelCounts[currentLabel] += 1  # 如果被记录过就+1
    shannonEnt = 0.0  # 香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 每个标签出现的频率
        shannonEnt -= prob * log(prob, 2)  # 熵 = sigma[(所有标签)=>(-1)*频率*log2(频率)]
    return shannonEnt


# 提取axis轴数据为value的向量
def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 使用拷贝建立新的划分列表
    for featVec in dataSet:  # 对于原数据的所有向量
        if featVec[axis] == value:  # 如果向量的axis维度是所需要的划分标签
            reducedFeatVec = featVec[:axis]  # 拷贝featVec
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择数据增益最大的特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征个数  len-1个特征，一个标签
    baseEntropy = calcShannonEnt(dataSet)  # 基准熵(不使用任何划分时数据的混乱程度)
    bestInfoGain = 0.0  # 最大信息增益
    bestFeature = -1  # 最大信息增益对应的特征
    for i in range(numFeatures):  # 对于所有的特征
        featList = [e[i] for e in dataSet]  # 从所有数据向量中抽出第i轴的数据
        uniqueVals = set(featList)  # 使用抽出的数据建立set，将某个特征的所有取值分离
        newEntropy = 0.0  # 使某种特征分割后的香农熵
        for value in uniqueVals:
            '''
            对于一组数据A,香农熵为ShA=ShannonEnt(A)
            按照某个特征分组，对于特征的n个取值分为n组
            n组分别记为S1 S2 ... Sn
            每组占总体数据的比例分别为p1=len(S1)/len(A) ... pn=len(Sn)/len(A) 
            每组对应香农熵为Sh1=ShannonEnt(S1) ... Shn=ShannonEnt(Sn)
            则划分后的香农熵总和为newSh=p1*Sh1 + p2*Sh2 + ... + pn*Shn
            划分后的信息增益为infoGain = ShA - newSh
            信息增益定为非负数，因为数据一旦被划分，一定变得有序
            信息增益越大，说明当前特征对数据起的决定性划分作用越强
            最大的信息增益所对应的特征，即为当前深度用于划分的分支
            '''
            subDataSet = splitDataSet(dataSet, i, value)  # 使用第i个特征的value取值划分数据
            prob = len(subDataSet) / float(len(dataSet))  # 这个数据子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 香农熵加权之和
        infoGain = baseEntropy - newEntropy  # 信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain  # 得到最大的信息增益
            bestFeature = i  # 与对应的特征
    return bestFeature  # 返回使信息增益最大的特征


# 多数表决来定义子节点的分类
def majorityCnt(classList):
    classCount = {}  # 字典 类别->个数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 按照个数排序，逆序，选出最大的
    return sortedClassCount[0][0]  # 返回个数最多的(第0个)元素的类别


# 建立树
def createTree(dataSet, labels):
    classList = [e[-1] for e in dataSet]  # 获得所有类标签
    if classList.count(classList[0]) == len(classList):  # 终止情况1:所有类标签相同
        return classList[0]  # 直接返回这些相同的标签
    if len(dataSet[0]) == 1:  # 数据集里每个数据向量只剩下一个数据，也就是标签，说明此时无法再分，使用多数表决返回
        return majorityCnt(classList)  # 多数表决
    # 如果不满足上面两个终止条件，继续选择最好特征来分割
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择使信息熵增益最大的特征
    bestFeatLabel = labels[bestFeat]  # 得到对应的label
    myTree = {bestFeatLabel: {}}  # 使用这个label建立字典，key是label名
    del (labels[bestFeat])  # 与dataSet变化相同，删除这个特征
    featValues = [e[bestFeat] for e in dataSet]  # 在dataSet中获得这个特征的所有取值
    uniqueVals = set(featValues)  # 独一无二化
    for value in uniqueVals:  # 对于所有取值
        subLabels = labels[:]  # 拷贝一份类标签，否则由于引用会造成数据紊乱
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        '''
        对于刚才（只有key）的字典，添加它的子树，子树就是代码最后的return myTree，使用递归造树
        '''
    return myTree


# 生成数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
