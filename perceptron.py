
import numpy as np
from math import copysign
import sklearn.linear_model
from enum import Enum
from numpy import linalg

class Kernel(Enum):
    LINEAR = 1
    LAPLACIANRBF = 2
    GAUSSIANRBF = 3
    POLYNOMIAL = 4


class Result():

    def __init__(self, coef, bias, converge):
        self.coef = coef
        self.bias = bias
        self.converge = converge


class KernelPerceptron():

    def __init__(self, x, y, kernel=Kernel.LINEAR, n_iter=20, p=3, sigma=5.0):
        self.data = x
        self.label = y
        self.kernel = kernel
        self.n_iter = n_iter
        self.alpha = None
        self.sigma = sigma
        self.p = p


    def kernel_func(self, xi, xj):

        def linear_kernel(x1, x2):
            return np.dot(x1, x2)

        def polynomial_kernel(x, y):
            return (1 + np.dot(x, y)) ** self.p

        def gaussian_kernel(x, y):
            return np.exp(-linalg.norm(x-y)**2 / (2 * (self.sigma ** 2)))

        if self.kernel == Kernel.LINEAR:
            return linear_kernel(xi, xj)
        elif self.kernel == Kernel.POLYNOMIAL:
            return polynomial_kernel(xi, xj)
        elif self.kernel == Kernel.GAUSSIANRBF:
            return gaussian_kernel(xi, xj)
        

    def train(self):
        alpha = np.zeros(len(self.data[0]))

        n_data, n_feature = self.data.shape
        alpha = np.zeros(n_data, dtype=np.float64)

        kernel_rst = np.zeros((n_data, n_data))

        for i in range(n_data):
            for j in range(n_data):
                # print self.data[i, :], self.data[j, :]
                kernel_rst[i, j] = self.kernel_func(self.data[i, :], self.data[j, :])

        for t in range(self.n_iter):
            for i in range(n_data):
                if np.sign(np.sum(kernel_rst[:, i] * alpha * self.label)) != self.label[i]:
                    alpha[i] += 1
        self.alpha = alpha
        support_vector = self.alpha > 1e-5
        self.alpha = self.alpha[support_vector]
        self.support_vector = self.data[support_vector]
        self.support_vector_y = self.label[support_vector]
        print len(self.alpha)

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            rst = 0
            for a, support_vector_y, support_vector in zip(self.alpha, self.support_vector_y, self.support_vector):
                rst += a * support_vector_y * self.kernel_func(X[i], support_vector)
            y_predict[i] = rst
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(self.project(X))


class Perceptron():

    def __init__(self, x, y, n_iter=20):
        self.data = x
        self.label = y
        self.n_iter = n_iter

    def train(self):
        def compute(xi, w, b):
            rst = np.dot(xi, w) + b
            return copysign(1, rst)
        n_data, n_feature = self.data.shape
        count = 0
        converge = False
        weights = np.zeros(n_feature)
        bias = 0
        while not converge and count < self.n_iter:
            converge = True
            updates = 0
            for i in range(len(self.data)):
                label_new = compute(self.data[i, :], weights, bias)
                if label_new * self.label[i] <= 0:
                    # print weights, label_new, self.label[i], self.data[i, :]
                    weights = weights + self.label[i] * self.data[i, :]
                    bias = bias + self.label[i]
                    converge = False
                    updates += 1
            count += 1
            print 'updates: ', str(updates), 'at interation: ', count
        print weights, converge, count, bias
        print map(lambda x: x / -bias, weights)
        self.weights = weights
        self.bias = bias
        self.converge = converge

    def project(self, data):
        return np.dot(data, self.weights) + self.bias

    def predict(self, data):
        data = np.atleast_2d(data)
        return np.sign(self.project(data))


def readfile(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    x = []
    y = []
    for l in lines:
        try:
            l = map(float, l.split())
            x.append(l[:-1])
            y.append(l[-1])
        except:
            pass

    print x, y

    return np.array(x), np.array(y)



def main():
    x, y = readfile('./a2_datasets/perceptron/percep2.txt')
    # p = Perceptron(x, y)
    # p.train()
    # r = p.predict(x)
    # print np.sum(r == y)

    # p = sklearn.linear_model.Perceptron(n_iter=10, shuffle=False)

    # p.fit(x, y)
    # print p.coef_

    p = KernelPerceptron(x, y, kernel=Kernel.GAUSSIANRBF)
    p.train()
    r = p.predict(x)
    print np.sum(r == y), 'total: ', len(x)

if __name__ == '__main__':
    main()