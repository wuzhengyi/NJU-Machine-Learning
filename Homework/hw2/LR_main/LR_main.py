import numpy

def loadTrainDataSet():
    '创建训练数据集，分类标签集并返回。'
    dataMat = []
    labelMat = []
    dataMat = numpy.genfromtxt("./assign2_dataset/page_blocks_train_feature.txt")
    labelMat = numpy.genfromtxt("./assign2_dataset/page_blocks_train_label.txt")
    # 线性归一化
    for i in range(dataMat.shape[1]):
        mins = numpy.min(dataMat[:, i])
        maxs = numpy.max(dataMat[:, i])
        for j in range(dataMat.shape[0]):
            dataMat[j][i] = (dataMat[j][i] - mins) / (maxs - mins)
    ones = numpy.ones(dataMat.shape[0])
    dataMat = numpy.c_[dataMat, ones]
    return dataMat, labelMat

def loadTestDataSet():
    '创建训练数据集，分类标签集并返回。'
    dataMat = []
    labelMat = []
    dataMat = numpy.genfromtxt("./assign2_dataset/page_blocks_test_feature.txt")
    labelMat = numpy.genfromtxt("./assign2_dataset/page_blocks_test_label.txt")
    # 线性归一化
    for i in range(dataMat.shape[1]):
        mins = numpy.min(dataMat[:, i])
        maxs = numpy.max(dataMat[:, i])
        for j in range(dataMat.shape[0]):
            dataMat[j][i] = (dataMat[j][i] - mins) / (maxs - mins)
    ones = numpy.ones(dataMat.shape[0])
    dataMat = numpy.c_[dataMat, ones]
    return dataMat, labelMat

# =====================================
# 输入：
#        x: w^Tx+b
# 输出:
#       p1
# =====================================

def p1(x):
    val = numpy.exp(-x)
    return 1.0 / (1 + val)


def firstDerivative(weight, dataMatrix, labelMat):
    '计算weight的一阶导数'

    # 初始化结果矩阵11*1
    result = numpy.zeros((dataMatrix.shape[1], 1))
    for i in range(dataMatrix.shape[0]):
        result -= dataMatrix[i].transpose() * (labelMat.A1[i] - p1(dataMatrix[i] * weight).A1[0])
    return numpy.mat(result)


def secondDerivative(weight, dataMatrix):
    '计算weight的二阶导数'

    # 初始化结果矩阵11*1
    result = numpy.zeros((dataMatrix.shape[1], dataMatrix.shape[1]))
    for i in range(dataMatrix.shape[0]):
        result += dataMatrix[i].transpose() * dataMatrix[i] * p1(dataMatrix[i] * weight).A1[0] * (1 - p1(dataMatrix[i] * weight).A1[0])
    return numpy.mat(result)


# =====================================
# 输入：
#        dataMatIn: 数据集
#        classLabels: 分类标签集
# 输出:
#        weights: 最佳拟合参数向量
# =====================================
def newton(dataMatIn, classLabels):
    '基于牛顿法的logistic回归分类器'

    # 将数据集，分类标签集存入矩阵类型。
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()

    # 初始化回归参数向量
    m, n = numpy.shape(dataMatrix)
    weights = numpy.mat(numpy.zeros((n, 1)))
    dert = numpy.inf

    # 对回归系数进行牛顿迭代
    while dert > numpy.exp(-3):
        f = firstDerivative(weights,dataMatrix,labelMat)
        s = secondDerivative(weights, dataMatrix)
        tempWeights = weights - numpy.linalg.pinv(s) * f
        dert = abs(sum((tempWeights - weights).A1))
        weights = tempWeights
    return weights


def gradAscent(dataMatIn, classLabels):
    '基于梯度上升法的logistic回归分类器'

    # 将数据集，分类标签集存入矩阵类型。
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()

    # 上升步长度
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # 初始化回归参数向量
    m, n = numpy.shape(dataMatrix)
    weights = numpy.ones((n, 1))

    # 对回归系数进行maxCycles次梯度上升
    for k in range(maxCycles):
        h = p1(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights

def test(num):
    '测试'
    dataArr, labelMat = loadTrainDataSet()
    trainLabel1 = labelMat
    t = 0; f = 0
    for index, value in enumerate(trainLabel1):
        if value != num:
            trainLabel1[index] = 0
            f+=1
        else:
            t+=1
    r = t/f
    # print(r)
    weights = newton(dataArr, labelMat)
    tp = 0; fp = 0; fn = 0; tn = 0;
    for i in range(dataArr.shape[0]):
        if p1(dataArr[i] * weights) > 0.5:
            if labelMat[i] == num:
                tp += 1
            else:
                fp += 1
        else:
            if labelMat[i]==num:
                fn += 1
            else:
                tn += 1
    print(tp,fn,fp,tn)
    print("正例：", num, "查准率：", tp/(tp+fp), "查全率：", tp/(tp+fn))



def newTest():
    trainFeature=numpy.genfromtxt("./testSet.txt")
    trainLabel=trainFeature[:,2]
    trainFeature=trainFeature[:,0:-1]
    for i in range(trainFeature.shape[1]):
        mins = numpy.min(trainFeature[:, i])
        maxs = numpy.max(trainFeature[:, i])
        for j in range(trainFeature.shape[0]):
            trainFeature[j][i] = (trainFeature[j][i]-mins)/(maxs-mins)
    ones = numpy.ones(trainFeature.shape[0])
    dataMat = numpy.c_[trainFeature, ones]
    weights = newton(dataMat, trainLabel)
    print(weights)

if __name__ == '__main__':
    test(1)
    test(2)
    test(3)
    test(4)
    test(5)
