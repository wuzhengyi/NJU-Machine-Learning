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


def p1(x):
    return 1.0 / (1 + numpy.exp(-x))


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
        result += dataMatrix[i].transpose() * dataMatrix[i] * p1(dataMatrix[i] * weight).A1[0] * (
                    1 - p1(dataMatrix[i] * weight).A1[0])
    return numpy.mat(result)


# =====================================
# 输入：
#        dataMatIn: 数据集
#        classLabels: 分类标签集
# 输出:
#        weights: 最佳拟合参数向量
# =====================================
def newton(trainFeature, trainLabel):
    '基于牛顿法的logistic回归分类器'

    # 将数据集，分类标签集存入矩阵类型。
    feature = numpy.mat(trainFeature)
    label = numpy.mat(trainLabel).transpose()

    # 初始化回归参数向量
    m, n = numpy.shape(feature)
    weights = numpy.mat(numpy.zeros((n, 1)))
    dert = numpy.inf

    # 对回归系数进行牛顿迭代
    tempWeights = numpy.mat(numpy.ones((n, 1)))
    times = 0
    # while dert > numpy.exp(-10):
    while dist(weights, tempWeights) > numpy.exp(-16):
    # while times < 20:
        times += 1
        f = firstDerivative(weights, feature, label)
        s = secondDerivative(weights, feature)
        tempWeights = weights
        # tempWeights = weights - numpy.linalg.pinv(s) * f
        # dert = abs(sum((tempWeights - weights).A1))
        weights = weights - numpy.linalg.pinv(s) * f
    print(times)
    return weights


def gradAscent(trainFeature, trainLabel):
    feature = numpy.mat(trainFeature)
    label = numpy.mat(trainLabel).transpose()
    weights = numpy.ones((feature.shape[1], 1))

    for k in range(100):
        h = p1(feature * weights)
        error = (label - h)
        weights = weights + 0.01 * feature.transpose() * error

    return weights


def dist(vec1, vec2):
    dist = numpy.linalg.norm(vec1 - vec2)
    return dist

def classify(num):

    dataArr, labelMat = loadTrainDataSet()
    trainLabel1 = labelMat

    for index, value in enumerate(trainLabel1):
        if value != num:
            trainLabel1[index] = 0

    # weights = newton(dataArr, labelMat)
    weights = gradAscent(dataArr, labelMat)

    tp = 0;
    fp = 0;
    fn = 0;
    tn = 0;
    for i in range(dataArr.shape[0]):
        if p1(dataArr[i] * weights) > 0.5:
            if labelMat[i] == num:
                tp += 1
            else:
                fp += 1
        else:
            if labelMat[i] == num:
                fn += 1
            else:
                tn += 1
    # print(tp, fn, fp, tn)
    # print("正例：", num, "查准率：", tp / (tp + fp), "查全率：", tp / (tp + fn))
    return weights

if __name__ == '__main__':

    feature, label = loadTestDataSet()
    label_dict = {}
    for l in label:
        if l not in label_dict:
            label_dict[l] = 1
        else:
            label_dict[l] += 1
    numLabel = label_dict.__len__()
    w = []
    for i in range(numLabel):
        w.append(classify(i + 1))
    # with open("myLabel.txt") as f:
    t = 0
    for i in range(feature.shape[0]):
        p = -numpy.inf
        myLabel = -1
        for j in range(numLabel):
            q = p1(feature[i] * w[j])
            print(j, q)
            if p < q:
                myLabel = j + 1
                p = q
        if label[i] == myLabel:
            t += 1
            print(myLabel)
    print(t, t/feature.shape[0])


