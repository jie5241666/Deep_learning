import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# plt.imshow(train_set_x_orig[28])
# plt.show()

#训练集，测试集样本数量与图像宽度
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print(train_set_x_orig.shape)

# 将每个样本展成一列
train_set_x_flatten = train_set_x_orig.reshape(m_train,-1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test,-1).T
print(train_set_x_flatten.shape)

# 标准化数据集
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def initialize_width_zeros(dim):
    """
    初始化参数为零
    :param dim:
    :return:
    """
    w = np.zeros((dim,1))
    b = 0
    return w,b
def propagate(w,b,X,Y):
    """
    前向传播
    :param w: 参数
    :param b: 偏置
    :param X: 训练数据
    :param Y: 标签
    :return:
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X))
    cost = -np.mean(Y*np.log(A)+(1-Y)*np.log(1-A))
    dw = (np.dot(X,(A-Y).T))/m
    db = np.mean(A-Y)
    grads = {"dw":dw,"db":db}
    return grads,cost

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    """
    优化参数w与b
    :param w:权重
    :param b:偏置
    :param X:训练集
    :param Y:标签
    :param num_iterations:迭代次数
    :param learning_rate:学习速率
    :param print_cost:是否每迭代100次打印一次损失值
    :return:
    """
    costs = []
    for i in range (num_iterations):
        # 计算代价值与梯度
        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        # 更新参数
        w = w-learning_rate*dw
        b = b-learning_rate*db
        #记录代价值
        if i%100 ==0:
            costs.append(cost)
        if print_cost and i%100==0:
            print("迭代{0}次后代价值为{1}".format(i,cost))
    params = {"w":w,"b":b}
    grads = {"dw":dw,"db":db}
    return params,grads,costs

def predict(w,b,X):
    """
    :param w: 权重
    :param b: 偏置
    :param X:训练集
    :return:
    """
    m = X.shape[1]
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)
    A[A>0.5] = 1
    A[A<=0.5] = 0
    return A

def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate = 0.5,print_cost = False):
    w,b = initialize_width_zeros(X_train.shape[0])
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    print("训练集精度为{0}%".format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("测试集精度为{0}%".format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))
    d = {
        "costs":costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train" : Y_prediction_train,
        "w" : w,
        "b" : b,
        "learning_rate" : learning_rate,
        "num_iterations": num_iterations
    }
    return d
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
