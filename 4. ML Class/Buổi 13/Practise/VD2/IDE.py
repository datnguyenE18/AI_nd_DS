import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import cv2

def showDigit(digits, i):
    print(digits.data[i])
    m = np.reshape(digits.data[i]/16, (8, 8))
    m2 = cv2.resize(m, (320, 320), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("digit", m2)
    cv2.waitKey(0)

def solve():
    digits = load_digits()
    # Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
    print('Image Data Shape', digits.data.shape)
    # Print to show there are 1797 labels (integers from 0–9)
    print("Label Data Shape", digits.target.shape)
    print(digits.data[0])

    showDigit(digits, 0)
    showDigit(digits, 9)
    showDigit(digits, 5)
    showDigit(digits, 3)

    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

    regr = LogisticRegression()
    regr.fit(X_train, y_train)
    print(regr.coef_, regr.intercept_)

    disp = metrics.plot_confusion_matrix(regr, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()

solve()