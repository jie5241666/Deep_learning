import numpy as np
import matplotlib.pyplot as plt
from testCases_v3 import *
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward

plt.rcParams['figure.figsize'] = (5.0,4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    parameters = {
        "W1":W1,
        "b2":b2,
        "W2":W2,
        "b2":b2
    }
    return parameters

def initialize_parameters_deep(lay_dims):
    parameters = {}
    L = len(lay_dims)
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(lay_dims[l],lay_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((lay_dims[l],1))
    return parameters

def linear_forward(A,W,b):
    Z = W.dot(A)+b
    cache = (A,W,b)
    return Z,cache
def linear_activation_forward(A_prev,W,b,activation):
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)
    cache = (linear_cache,activation_cache)
    return A,cache

def L_model_forward(X,parameters):
    caches = []
    A = X
    L = len(parameters)//2
    for l in range(1,L):
        # A_prev = A
        A,cache = linear_activation_forward(A,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)
    AL,cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    return AL,caches

def compute_cost(AL,Y):
    cost = -np.mean(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    return cost
def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m = A_prev.shape[1]
    dW = dZ.dot(A_prev.T)/m
    db = (np.mean(dZ,axis=1).reshape(dZ.shape[0],1))
    dA_prev = (W.T).dot(dZ)
    return dA_prev,dW,db
def linear_activation_backward(dA, cache, activation):
    linear_cache,activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    return dA_prev,dW,db


def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    current_cache = caches[L-1]
    grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL,current_cache,activation="sigmoid")
    for l in reversed(range(L-1)):
        current_cache  = caches[l]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
    return grads
def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)//2
    for l in range(L):
        parameters["W"+str(l+1)] -= learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] -= learning_rate*grads["db"+str(l+1)]
    return parameters
