from os import listdir
from numpy import *
import operator
from kNN import classify0


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("trainingDigits/%s" % fileNameStr)
        testFileList = listdir("testDigits")
        errorCount = 0.0
        mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("came back with %d, real is %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
            print("came back with %d, real is %d" % (classifierResult, classNumStr), end="  ")
            print("file name : %s " % fileNameStr)

    print("\n number of error : %d" % errorCount)
    print("\n error rate : %f " % (errorCount / float(mTest)))
    print("\n acc rate : %f" % (1.0 - (errorCount / float(mTest))))


if __name__ == "__main__":
    handwritingClassTest()
