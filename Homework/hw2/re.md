设$w_1 - w_{10},b =0$，应用牛顿迭代   
定义两个函数分别求一阶导数与二阶导数 
```python
def p1(x):
    return 1.0 / (1 + numpy.exp(-x))

# 一阶导数
def firstDerivative(weight, trainData, trainLabel):
    result = Matrix zeros (11*1)
    for x in trainData:
        result -= x.transpose * ( trainLabel[i] - p1(x*weight))
    return result

# 二阶导数
def secondDerivative(weight, trainData):
    result = Matrix zeros (11*11)
    for x in trainData:
        result += x.transpose * x * p1(x * weight) * (1 - p1(x * weight))
    return result

# 牛顿迭代
def newton(trainData,trainLabel):
    w = Matrix zeros (11*1)
    while( dertw > 1e-3):
        w = w - secondDerivative(w, trainData).transpose * firstDerivative(w, trainData, trainLabel)
    return w
```