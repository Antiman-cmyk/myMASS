import numpy as np
import matplotlib.pyplot as plt
import random

'''
def gen_linear_data(w, n):
    num_feature = len(w)
    dataset = np.zeros([n, num_feature+1])

    for i in range(n):
        x = np.random.rand(1, num_feature)*20 - 10
        inner_dot = np.sum(w * x)
        if inner_dot > 0:
            dataset[i] = np.append(x, 1)
        else:
            dataset[i] = np.append(x, -1)
    return dataset
'''


# 生成非线性数据
def gen_nonlinear_data(num):
    result = []
    g1 = [random.random() * 20, random.random() * 20]
    g2 = [random.random() * 20, random.random() * 20]
    # 由数据范围内的两个点来确定分割线，保证划分线一定会经过生成的点的范围
    w = [(g1[1] - g2[1]) / (g1[0] - g2[0]), -1, g1[1] - (g1[1] - g2[1]) / (g1[0] - g2[0]) * g1[0]]

    result.append(w)
    for i in range(num - 4):
        x = [random.random() * 20, random.random() * 20]
        y = w[0] * x[0] + w[1] * x[1] + w[2]
        if y < 0:
            x.append(-1)
        elif y > 0:
            x.append(1)
        else:
            continue
        result.append(x)
    for i in range(num - 4, num):
        x = [random.random() * 20, random.random() * 20]
        y = w[0] * x[0] + w[1] * x[1] + w[2]
        if y < -1:
            x.append(1)
        elif y > 1:
            x.append(1)
        else:
            continue
        result.append(x)
    # result[0] 是 w, 后面的是x[横坐标，纵坐标，标签]
    return result


'''
#计算错误率
def checkErrorRate(test,w):
    count=0
    for i in range(len(test)):
        x=np.array(test[i][:-1])
        y=np.dot(x,w)
        if np.sign(test[i][-1])!=np.sign(y):
            count+=1
    return count/len(test)
#pocket算法实现
def Pocket():
    w=np.zeros(3)#初始化w0
    best_w=w
    bestRate=1
    n=50
    m=50
    c1,c2=generate_data(1,1,-1,1,n,m)
    test=np.vstack((c1,c2))#合并两类数据
    x0 = np.ones(n+m)
    test = np.c_[x0, test]  # 插入列向量x0=[0,0,...0]
    cnt=0
    while True:
        cnt+=1
        if cnt>1000:#pocket与pla不同的一点就在于他靠控制迭代次数来提高分类精度
            break
        success=True
        for i in range(len(test)):
            x=np.array(test[i][:-1])
            y=np.dot(x,w)
            if np.sign(y)==np.sign(test[i][-1]):
                continue
            w=w+test[i][-1]*x #更新w值
            rate=checkErrorRate(test,w)#得出分错率
            if rate<bestRate:#如果分错率更小则替换当前最好的w
                bestRate=rate
                best_w=w
            success = False
            break
        if success==True:
            break
'''


def pocket(data):
    num_fault_list = []
    w_list = []
    X = []
    y = []
    for i in range(1, len(data)):  # data[0]是w
        X.append((data[i][0], data[i][1]))  # x [横，纵]
        y.append(data[i][2])  # label
    X = np.array(X)
    y = np.array(y)
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # 补x0=1便于后续计算
    w = np.random.randn(3, 1)  # 初始化w权重数组

    for i in range(100000):
        s = np.dot(X, w)
        y_pred = np.ones_like(y)
        loc_n = np.where(s < 0)[0]
        y_pred[loc_n] = -1
        num_fault = len(np.where(y != y_pred)[0])
        num_fault_list.append(num_fault)
        # print("%d: Wrong Point: %d"%(i, num_fault))
        if num_fault == 0:
            w_list.append(w)
            break
        else:
            t = random.choice(np.where(y != y_pred)[0])
            w = w + y[t] * X[t, :].reshape((3, 1))
            w_list.append(w)
    print(num_fault_list)
    index_min = num_fault_list.index(min(num_fault_list))
    print(index_min)
    print(num_fault_list[index_min])
    w = w_list[index_min]
    print(w)
    return w


# 可视化
def VisualizePoint(data, w, w2):
    x = np.linspace(0, 20, 50)  # 在1到20之间产生50组数据(数据之间呈等差数列)
    if len(w) != 0:
        z = - w[0] / w[2] - w[1] / w[2] * x
        plt.plot(x, z, color="orange", linestyle="--")
    if len(w2) != 0:
        z = - w2[2] / w2[1] - w2[0] / w2[1] * x
        plt.plot(x, z, color="blue", linestyle="--")
    posx = []
    posy = []
    negx = []
    negy = []
    for i in range(1, len(data)):
        if data[i][-1] == -1:
            negx.append(data[i][0])
            negy.append(data[i][1])
        else:
            posx.append(data[i][0])
            posy.append(data[i][1])
    plt.scatter(negx, negy, marker='x', c='r', label="+1")
    plt.scatter(posx, posy, marker='o', c='g', label="-1")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="upper left")
    plt.title("Original Data")
    # plt.xlim(0,20)
    # plt.ylim(0,20)
    plt.show()


if __name__ == "__main__":
    all = gen_nonlinear_data(20)
    # print(all)
    w = pocket(all)
    VisualizePoint(all, w, all[0])
