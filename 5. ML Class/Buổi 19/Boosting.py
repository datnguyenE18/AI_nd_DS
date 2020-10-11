import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from LogisticRegression import read_ex2data1

from sklearn.datasets import load_digits
def test1():
    X, Y = read_ex2data1()
    # print(X.shape,Y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    # print(X_test)

    clf = AdaBoostClassifier()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)

    disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()

def test2():
    digits = load_digits()

    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=600,
        algorithm="SAMME")

    clf.fit(X_train, y_train)

    errors = []

    for y_pred in clf.staged_predict(X_test):
        errors.append(
            1. - metrics.accuracy_score(y_pred, y_test))

    n_trees = len(clf)

    estimator_errors = clf.estimator_errors_[:n_trees]
    estimator_weights = clf.estimator_weights_[:n_trees]

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(range(1, n_trees + 1),
             errors, c='black')
    plt.legend()
    plt.ylim(0.18, 0.62)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Trees')

    plt.subplot(132)
    plt.plot(range(1, n_trees + 1), estimator_errors,
             "b", alpha=.5)

    plt.ylabel('Error')
    plt.xlabel('Number of Trees')
    plt.ylim((.2,
              estimator_errors.max() * 1.2))
    plt.xlim((-20, len(clf) + 20))

    plt.subplot(133)
    plt.plot(range(1, n_trees + 1), estimator_weights,
             "b")
    plt.legend()
    plt.ylabel('Weight')
    plt.xlabel('Number of Trees')
    plt.ylim((0, estimator_weights.max() * 1.2))
    plt.xlim((-20, n_trees + 20))

    # prevent overlapping y-axis labels
    plt.subplots_adjust(wspace=0.25)
    plt.show()

def test():
    #test1()
    test2()