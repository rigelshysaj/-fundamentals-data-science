import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    # return the sigmoid of x
    g = 1 / ( 1 + np.exp(-x))
    return g

def log_likelihood(theta, x, y):
    # return the log likehood of theta according to data x and label y
    m = x.shape[0]
    h = sigmoid(np.dot(x, theta))
    log_l =  ( np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)) ) / m
    return log_l

def grad_l(theta, x, y):
    # return the gradient G of the log likelihood
    #m = x.shape[0]
    h = sigmoid(np.dot(x, theta))
    G = np.dot(np.transpose(x), y - h)
    return G

def gradient_ascent(theta, x, y, G, alpha=0.01, iterations=100):
    m = len(y)
    log_l_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, x.shape[1]))

    # return the optimized theta parameters,
    # as well as two lists containing the log likelihood's and values of theta at all iterations
    for i in range(iterations):
        theta = theta + alpha * G(theta, x, y) / m
        theta = np.array(theta, dtype=np.float32)
        theta_history[i] = theta
        log_l_history[i] = log_likelihood(theta, x, y)
    return theta, log_l_history, theta_history

def rescale(data: np.array):
    """
    Rescales data into [0, 1] range
    """
    for i in range(data.shape[1]):
        column = data[:, i]
        column = (column - column.min()) / (column.max() - column.min())
        data[:, i] = column

def plot_rpc(predictions, labels, plot=False):
    recall = []
    precision = []

    sortidx = predictions.argsort()
    p = predictions[sortidx]
    l = labels[sortidx]

    tn = 0
    fn = 0
    tp = len(p) - len(np.where(l == 0)[0])
    fp = len(p) - tp
    for i in range(len(predictions)):
        if l[i] == 1:
            tp -= 1
            fn += 1
        else:
            tn += 1
            fp -= 1

        #Compute precision and recall values and append them to "recall" and "precision" vectors
        if tp + fp != 0:
            precision += [tp / (tp + fp)]
        else: 
            precision += [1.0]
        recall += [tp / (tp + fn)]
    
    if plot:
        plt.plot([1-precision[i] for i in range(len(precision))], recall)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('1 - precision')
        plt.ylabel('recall')
        plt.show()
    return [1-precision[i] for i in range(len(precision))], recall

def hess_l(theta, x, y):
    # return the Hessian matrix hess
    # m = len(x)
    h = sigmoid(np.dot(x, theta))
    hess = np.dot(np.dot(x.T, np.diag(h * (h - 1))), x) # / m
    return hess

def newton(theta0, x, y, G, H, eps):
    # return the optimized theta parameters,
    # as well as two lists containing the log likelihood's and values of theta at all iterations
    cur_theta = theta0
    next_theta = None
    theta_history = []
    log_l_history = []
    for _ in range(1000):
        hess = H(cur_theta, x, y)
        grad = G(cur_theta, x, y) / len(x)
        next_theta = cur_theta - np.linalg.inv(hess).dot(grad)
        theta_history.append(cur_theta)
        log_l_history.append(log_likelihood(cur_theta, x, y))
        cur_theta = next_theta
        if abs(np.linalg.norm(grad)) < eps:
            break

    theta = next_theta
    return theta, theta_history, log_l_history

def plot_all_rpc(to_plot: list):
    colors = ["b", "r", "g"]
    styles = [":", "--", "-."]
    for x, y, name in to_plot:
        precision, recall = plot_rpc(x, y)
        plt.plot(precision, recall, label=name, linestyle = styles.pop(), color=colors.pop())
    plt.title("Precision/Recall curves comparison")
    plt.legend(loc="lower right")
    plt.axis([0, 1, 0, 1])
    plt.xlabel('1 - precision')
    plt.ylabel('recall')
    plt.show()