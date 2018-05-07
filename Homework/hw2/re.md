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

步长0.01， 迭代1000次

过采样
真正例数量:453	准确率:0.842007
标记2	查准率:0.574074	查全率:0.837838
标记5	查准率:0.140000	查全率:0.583333
标记1	查准率:0.980815	查全率:0.848548
标记4	查准率:0.266667	查全率:1.000000
标记3	查准率:1.000000	查全率:0.666667

无过采样
真正例数量:509	准确率:0.946097
标记1	查准率:0.965164	查全率:0.977178
标记2	查准率:0.769231	查全率:0.810811
标记3	查准率:1.000000	查全率:0.666667
标记4	查准率:1.000000	查全率:0.750000
标记5	查准率:0.500000	查全率:0.250000

步长0.01， 迭代500次
过采样
真正例数量:489	准确率:0.908922
标记4	查准率:0.400000	查全率:1.000000
标记1	查准率:0.973970	查全率:0.931535
标记5	查准率:0.210526	查全率:0.333333
标记2	查准率:0.966667	查全率:0.783784
标记3	查准率:0.166667	查全率:1.000000