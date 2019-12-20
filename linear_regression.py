import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def hypothesis(X, w):
    return np.dot(X, w)

def calculate_cost(y, h):
    m = np.shape(y)[0]
    return np.sum(np.power(y - h, 2)) / m

def fit(X, y, leaning_rate=0.001, iterations=100):
    m, n = np.shape(X)
    ones = np.ones((m, 1))
    X = np.concatenate((ones, X), axis=1)
    w_init = np.random.rand(1, n + 1).T
    w = [w_init]
    cost_histories = []
    for iteration in range(iterations):
        h = hypothesis(X, w[-1])
        loss = h - y
        gradient = leaning_rate * np.dot(X.T, loss) / m
        w_new = w[-1] - gradient
        cost = calculate_cost(y, h)
        cost_histories.append(cost)
        w.append(w_new)
    return w, cost_histories    

def draw_cost(cost_histories, iterations):
    plt.xlabel('x')
    plt.ylabel('y')
    iterations = len(cost_histories)
    plt.plot(range(iterations), cost_histories, 'b.')
    plt.show()

