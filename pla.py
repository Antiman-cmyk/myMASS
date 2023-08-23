from numpy import *
import matplotlib.pyplot as plt
import operator
import time
import random


def random_dot_label(left, right, n, train=True):
    left, right = (int(left), int(right)) if left <= right else (int(right), int(left))
    n = int(abs(n)) if n else 0
    # if train:
    #     return [[random.uniform(left, right) for _ in range(3)] for __ in range(n)], [random.choice([-1, 1]) for _ in range(n)]
    # else:
    #     return [[random.uniform(left, right) for _ in range(3)] for __ in range(n)]
    if train:
        return [[random.randint(left, right) for _ in range(3)] for __ in range(n)], [random.choice([-1, 1]) for _ in range(n)]
    else:
        return [[random.randint(left, right) for _ in range(3)] for __ in range(n)]

# def createTrainDataSet(left, right, num):  # 训练样本,第一个1为阈值对应的w，下同
#     # trainData = [[1, 1, 4],
#     #              [1, 2, 3],
#     #              [1, -2, 3],
#     #              [1, -2, 2],
#     #              [1, 0, 1],
#     #              [1, 1, 2]]
#     # label = [1, 1, 1, -1, -1, -1]
#     trainData, label = random_dot_label(-10, 10, 50)
#     return trainData, label
#
#
# def createTestDataSet():  # 数据样本
#     testData = [[1, 1, 1],
#                 [1, 2, 0],
#                 [1, 2, 4],
#                 [1, 1, 3]]
#     return testData


def sigmoid(X):
    X = float(X)
    if X > 0:
        return 1
    elif X < 0:
        return -1
    else:
        return 0


def pla(traindataIn, trainlabelIn):
    traindata = mat(traindataIn)
    trainlabel = mat(trainlabelIn).transpose()
    m, n = shape(traindata)
    w = ones((n, 1))
    while True:
        iscompleted = True
        for i in range(m):
            if (sigmoid(dot(traindata[i], w)) == trainlabel[i]):  # 矩阵乘
                continue
            else:
                iscompleted = False
                w += (trainlabel[i] * traindata[i]).transpose()  # 修正公式 w(t+1)=w(t) + y(t)x(t)
        if iscompleted:
            break
    return w


def classify(inX, w):
    result = sigmoid(sum(w * inX))
    if result > 0:
        return 1
    else:
        return -1


def plotBestFit(w, left, right, num):
    traindata, label = random_dot_label(left, right, num)
    dataArr = array(traindata)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')  # 绘制散点图 标记点是方形
    ax.scatter(xcord2, ycord2, s=30, c='green')  # 绘制散点图 标记点是圆圈
    x = arange(left, right, 0.1)
    y = (-w[0] - w[1] * x) / w[2]  # 公式来源：分类直线(x2即y) w0 + w1*x1 + w2*x2 = 0 变换后即 y = (-w[0]-w[1] * x)/w[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyall(datatest, w):
    predict = []
    for data in datatest:
        result = classify(data, w)
        predict.append(result)
    return predict


def main():
    left = -10.0
    right = 10.0
    num_train = 10
    num_test = 10
    trainData, label = random_dot_label(left, right, num_train)
    testdata = random_dot_label(left, right, num_test, train=False)
    w = pla(trainData, label)
    result = classifyall(testdata, w)
    plotBestFit(w, left, right, num_train)
    print(w)
    print(result)


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print('finish all in %s' % str(end - start))
