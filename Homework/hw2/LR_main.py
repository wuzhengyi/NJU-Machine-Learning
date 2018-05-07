import numpy
import random


def loadTrainDataSet(isOverSampling):
    '创建训练数据集，分类标签集并返回。'
    feature = []
    label = []
    feature = numpy.genfromtxt("./assign2_dataset/page_blocks_train_feature.txt")
    label = numpy.genfromtxt("./assign2_dataset/page_blocks_train_label.txt")
    # 过采样
    if isOverSampling:
        feature, label = overRandomSampling(feature, label)
    # 线性归一化
    for i in range(feature.shape[1]):
        mins = numpy.min(feature[:, i])
        maxs = numpy.max(feature[:, i])
        for j in range(feature.shape[0]):
            feature[j][i] = (feature[j][i] - mins) / (maxs - mins)
    ones = numpy.ones(feature.shape[0])
    feature = numpy.c_[feature, ones]

    return feature, label


def loadTestDataSet():
    '创建训练数据集，分类标签集并返回。'
    feature = []
    label = []
    feature = numpy.genfromtxt("./assign2_dataset/page_blocks_test_feature.txt")
    label = numpy.genfromtxt("./assign2_dataset/page_blocks_test_label.txt")
    # 线性归一化
    for i in range(feature.shape[1]):
        mins = numpy.min(feature[:, i])
        maxs = numpy.max(feature[:, i])
        for j in range(feature.shape[0]):
            feature[j][i] = (feature[j][i] - mins) / (maxs - mins)
    ones = numpy.ones(feature.shape[0])
    feature = numpy.c_[feature, ones]
    return feature, label


def randomData(feature, num):
    m, n = feature.shape
    newTable = numpy.zeros((num, n))

    for i in range(num):
        newTable[i, :] = feature[random.randint(0, m - 1), :]
    return newTable


def smoteData(feature, num):
    m, n = feature.shape
    # D = numpy.zeros((m, m))
    # for i in range(m):
    #     for j in range(i)
    #         D[i][j] = D[j][i] = dist(feature[i, :],feature[j, :])

    newTable = numpy.zeros((num, n))
    for i in range(num):
        src = feature[random.randint(0, m - 1), :]
        dest = feature[random.randint(0, m - 1), :]
        newTable[i, :] = src + random.random()*(dest - src)
    return newTable


# def overRandomSampling(feature, label):
#     table = numpy.c_[feature, label]
#     table = table[numpy.lexsort(table.T)]
#     label_dict = {}
#     for l in label:
#         if l not in label_dict:
#             label_dict[l] = 1
#         else:
#             label_dict[l] += 1
#     maxLabel = sorted(label_dict, key=lambda x: label_dict[x])[-1]
#
#     begin = sum(label<maxLabel);
#     end = sum(label<maxLabel + 1)
#     tempTable = feature[begin:end, :]
#     tempLabel = numpy.full((label_dict[maxLabel], 1), maxLabel)
#     for key in range(1, int(max(label_dict.keys())) + 1):
#         if key!=maxLabel:
#             begin = sum(label < key);
#             end = sum(label < key + 1)
#             oldFeature = feature[begin:end, :]
#             # newFeature = randomData(oldFeature,label_dict[maxLabel])
#             newFeature = smoteData(oldFeature, label_dict[maxLabel])
#             newLabel = numpy.full((label_dict[maxLabel], 1), key)
#             tempTable = numpy.concatenate((tempTable, newFeature), axis=0)
#             tempLabel = numpy.concatenate((tempLabel, newLabel), axis=0)
#     return tempTable, tempLabel


def overRandomSampling(feature, label):
    table = numpy.c_[feature, label]
    table = table[numpy.lexsort(table.T)]
    label_dict = {}
    for l in label:
        if l not in label_dict:
            label_dict[l] = 1
        else:
            label_dict[l] += 1
    maxLabel = sorted(label_dict, key=lambda x: label_dict[x])[-1]
    newFeature = numpy.zeros((label_dict[maxLabel] * numLabel, feature.shape[1]))
    newLabel = numpy.zeros(label_dict[maxLabel] * numLabel)
    up = 0
    # np.concatenate((a, b), axis=0)
    for key in range(1, int(max(label_dict.keys())) + 1):
        low = up
        up += label_dict[key]
        # print(label_dict[key], labelMax)
        if label_dict[key] == label_dict[maxLabel]:
            for i in range(label_dict[maxLabel]):
                newFeature[i + label_dict[maxLabel] * (key - 1), :] = table[i + low, 0:-1]
                newLabel[i + label_dict[maxLabel] * (key - 1)] = key
        else:
            for i in range(label_dict[maxLabel]):
                r = random.randint(0, label_dict[key]-1)
                newFeature[i + label_dict[maxLabel] * (key - 1), :] = table[r + low, 0:-1]
                newLabel[i + label_dict[maxLabel] * (key - 1)] = key

    return newFeature, newLabel


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
    while dist(weights, tempWeights) > numpy.exp(-1):
        f = firstDerivative(weights, feature, label)
        s = secondDerivative(weights, feature)
        tempWeights = weights
        weights = weights - numpy.linalg.pinv(s) * f
    return weights


def gradAscent(trainFeature, trainLabel):
    feature = numpy.mat(trainFeature)
    label = numpy.mat(trainLabel).transpose()
    weights = numpy.zeros((feature.shape[1], 1))

    for k in range(500):
        h = p1(feature * weights)
        error = (label - h)
        weights = weights + 0.01 * feature.transpose() * error

    return weights


def dist(vec1, vec2):
    dist = numpy.linalg.norm(vec1 - vec2)
    return dist


def classify(num, isOverSampling, isNewton, rescaling):

    trainFeature, trainLabel = loadTrainDataSet(isOverSampling)
    tempTrainLabel = trainLabel

    # print(trainLabel.shape,trainFeature.shape)
    pos = 0
    neg = 0
    for index, value in enumerate(tempTrainLabel):
        if value != num:
            tempTrainLabel[index] = 0
            neg += 1
        else:
            tempTrainLabel[index] = 1
            pos += 1
    # 牛顿迭代
    if isNewton:
        weights = newton(trainFeature, tempTrainLabel)
    # 梯度下降
    else:
        weights = gradAscent(trainFeature, tempTrainLabel)

    tp = 0;
    fp = 0;
    fn = 0;
    tn = 0;
    divition = 0;
    if rescaling:
        divition = pos/(pos + neg)
    else:
        divition = 0.5
    for i in range(trainFeature.shape[0]):
        if p1(trainFeature[i] * weights) > divition:
            if trainLabel[i] == num:
                tp += 1
            else:
                fp += 1
        else:
            if trainLabel[i] == num:
                fn += 1
            else:
                tn += 1
    # print(tp, fn, fp, tn)
    # print("正例：", num, "查准率：", tp / (tp + fp), "查全率：", tp / (tp + fn))
    return weights


if __name__ == '__main__':
    # 是否进行过采样
    isOverSampling = True
    # 方法选择梯度还是牛顿法
    isNewton = False
    # 在放缩
    rescaling = False
    feature, label = loadTestDataSet()
    label_dict = {} # 正例
    for l in label:
        if l not in label_dict:
            label_dict[l] = 1
        else:
            label_dict[l] += 1
    # print(label_dict)
    numLabel = label_dict.__len__()
    w = []
    for i in range(numLabel):
        w.append(classify(i + 1, isOverSampling, isNewton, rescaling))
        print("完成标记%d的计算" % (i))

    tp = {} # 真正例
    pt = {} # 预测的正例
    for i in range(feature.shape[0]):
        p = -numpy.inf
        myLabel = -1
        for j in range(numLabel):
            q = p1(feature[i] * w[j])
            if p < q:
                myLabel = j + 1
                p = q
        if myLabel not in pt:
            pt[myLabel] = 1
        else:
            pt[myLabel] += 1
        if label[i] == myLabel:
            if myLabel not in tp:
                tp[myLabel] = 1
            else:
                tp[myLabel] += 1
    t = 0 # 所有正确的数量
    for i in tp.values():
        t += i
    print("真正例数量:%d\t准确率:%f" % (t, t/feature.shape[0]))
    for key in pt:
        if key in tp:
            print("标记%d\t查准率:%f\t查全率:%f" % (key, tp[key]/pt[key], tp[key]/label_dict[key]))


