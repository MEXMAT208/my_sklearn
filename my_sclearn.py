import numpy as np
import matplotlib.pyplot as plt
import random
import math

class My_logistic_regression:
    def __init__(self, m):

        self.k = np.zeros((m, 1))
        self.b = 0

    def sigmoid(self, x):
        return 1./(1 + np.exp(-x))

    def train(self, X, Y, lr, num_epochs):

        X = np.array(X)
        Y = np.array(Y)

        losses = []

        N = len(X)

        for i in range(num_epochs):

            sigm_logits = self.predict(X)

            losses.append(self.log_loss(sigm_logits, Y))

            d_k = (X * (sigm_logits - Y)).sum(0).reshape(-1, 1) / N

            d_b = (sigm_logits - Y).sum() / N

            self.k -= lr * d_k
            self.b -= lr * d_b

        return losses

    def predict(self, X):

        N = len(X)

        logits = X @ self.k + np.full((N, 1), self.b)

        return self.sigmoid(logits)

    def predict_labels(self, X):

        def filt(x):

            if x >= 0.5:
                return 1
            else:
                return 0

        return np.array([filt(x[0]) for x in X]).reshape(-1, 1)


    def log_loss(self, Y, Y_labels):

        return -np.sum(Y_labels * np.log(Y) + (1 - Y_labels) * np.log(1 - Y)) / len(Y)


    def print_coeff(self):

        print('k: {}'.format(self.k.reshape(-1)))
        print('b: {}'.format(self.b))

    def metrics(self, Y_labels, Y_true):

        TP, TN, FP, FN = 0, 0, 0, 0

        for i in range(len(Y_labels)):

            if Y_labels[i] == 1:
                if Y_true[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if Y_true[i] == 1:
                    FN += 1
                else:
                    TN += 1

        precision = TP/(TP + FP)
        recall = TP/(TP + FN)

        f_measure = 2 * precision * recall / (precision + recall)

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('f-measure: {}'. format(f_measure))

    def roc_auc(self, Y_preds, Y_labels):

        a = zip(Y_preds, Y_labels)

        a = sorted(a, key = lambda x: x[0], reverse = True)

        zeros = 0
        bads = 0

        for x in a:

            if x[1] == 0:
                zeros += 1
            else:
                bads += zeros


        hm_pairs = zeros * (len(Y_preds) - zeros)

        return 1 - bads / hm_pairs

class My_linear_regression:
    def __init__(self, m):

        self.k = np.zeros((m, 1))
        self.b = 0

    def train(self, X, Y, lr, num_epochs):

        X = np.array(X)

        Y = np.array(Y)

        losses = []

        N = len(X)

        for i in range(num_epochs):

            logits = X @ self.k + np.full((N, 1), self.b) - Y

            loss = np.sum(np.square(logits)) / N

            losses.append(loss)

            d_b = 2 * np.sum(logits) / N
            d_k = 2 * (X * logits).sum(0).reshape(-1, 1) / N

            self.k -= lr * d_k
            self.b -= lr * d_b


        return losses

    def MSE(self, X, Y):

        N = len(X)

        logits = X @ self.k + np.full((N, 1), self.b) - Y

        return np.sum(np.square(logits)) / N

    def print_coeff(self):

        print('k: {}'.format(self.k.reshape(-1)))
        print('b: {}'.format(self.b))

class My_kNN_classification:
    def __init__(self, k, labels):
        self.k = k
        self.labels = labels
        
    def nearest(self, X, Y, data):
        
        X = np.array(X)
        Y = np.array(Y)
        data = np.array(data)
        
        answ = []
        
        for x in data:
        
            list_dist = np.square(X - x).sum(1)
            
            print(list_dist)

            list_dist = list(enumerate(list_dist))

            list_dist = sorted(list_dist, key = lambda p : p[1])[:self.k]

            list_dist = sorted([p[0] for p in list_dist])

            hm_labels = np.zeros(self.labels)

            for y in list_dist:
                hm_labels[Y[y]] += 1
                
            
            answ.append(hm_labels.argmax())
        
        return answ
                                      
    def accuracy(self, x, y):
        return np.sum(x == y) / len(x)

class My_kNN_regression:
    def __init__(self, k):
        self.k = k

    def nearest(self, X, Y, data):

        X = np.array(X)
        Y = np.array(Y)
        data = np.array(data)

        answ = []

        for x in data:

            list_dist = np.square(X - x).sum(1)

            list_dist = list(enumerate(list_dist))

            list_dist = sorted(list_dist, key = lambda p : p[1])[:self.k]

            list_dist = sorted([p[0] for p in list_dist])

            avg_sum = 0

            for y in list_dist:
                avg_sum += Y[y]


            answ.append(avg_sum / self.k)

        return answ

    def MSE(self, x, y):
        return np.sum(np.square(x - y)) / len(x)

