import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import  plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

X,Y = load_planar_dataset()
# Y = Y[0,:]
# plt.scatter(X[0,:],X[1,:],s=40,c=Y,cmap=plt.cm.Spectral)
# plt.show()

def layer_sizes(X,Y):
    n_x = (X.shape)[0]
    n_h = 4
    n_y = (Y.shape)[0]
    return (n_x,n_h,n_y)
def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    paramters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return paramters

def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = W1.dot(X)+b1
    A1 = np.tanh(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = sigmoid(Z2)
    cache = {"Z1":Z1,
             "A1":A1,
             "Z2":Z2,
             "A2":A2}
    return A2,cache

def compute_cost(A2,Y,parameters):
    cost = -np.mean(np.log(A2)*Y+(1-Y)*np.log(1-A2))
    return cost
def backward_propagation(parameters,cache,X,Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2-Y
    dW2 = dZ2.dot(A1.T)/m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1 = ((W2.T).dot(dZ2))*(1-np.power(A1,2))
    dW1 = dZ1.dot(X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    grads = {
        "dW1":dW1,
        "db1":db1,
        "dW2":dW2,
        "db2":db2
    }
    return grads

def update_parameters(parameters,grads,learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2,}
    return parameters

def nn_model(X,Y,n_h,num_iterations = 10000,print_cost = False):
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range (0,num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate=1.2)
        if print_cost and i%1000 ==0:
            print("迭代{0}次损失值为{1}".format(i,cost))
    return parameters

def predict (parameters,X):
    A2,cache = forward_propagation(X,parameters)
    A2[A2>0.5] = 1
    A2[A2<=0.5] = 0
    predictions = A2
    return predictions

parameters = nn_model(X,Y,n_h = 4,num_iterations=10000,print_cost=True)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0,:])
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
