import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

import cv2

##############################################################
def read_ex2data1():
    X = []
    Y = []
    with open('ex2data1.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                X.append([float(row['x1']), float(row['x2'])])
                Y.append(float(row['y']))

            line_count += 1
    return np.array(X), np.array(Y)
def test1():
    X, Y = read_ex2data1()
    #print(X.shape,Y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    #print(X_test)

    regr = LogisticRegression()

    regr.fit(X_train, y_train)
    print(regr.coef_, regr.intercept_)

    y_pred = regr.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)

    disp = metrics.plot_confusion_matrix(regr, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()

##############################################################
def showDigit(digits, i):
    print(digits.data[i])
    m = np.reshape(digits.data[i]/16, (8, 8))
    m2 = cv2.resize(m, (320, 320), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("digit", m2)
    cv2.waitKey(0)

def test2():
    digits = load_digits()
    # Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
    print('Image Data Shape', digits.data.shape)
    # Print to show there are 1797 labels (integers from 0–9)
    print("Label Data Shape", digits.target.shape)
    print(digits.data[0])

    showDigit(digits, 0)
    showDigit(digits, 1)
    showDigit(digits, 2)
    showDigit(digits, 3)

    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

    regr = LogisticRegression()
    regr.fit(X_train, y_train)
    print(regr.coef_, regr.intercept_)

    disp = metrics.plot_confusion_matrix(regr, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()

def test():
    #test1()
    test2()