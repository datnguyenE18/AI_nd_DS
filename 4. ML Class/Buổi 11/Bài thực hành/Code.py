import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def randomPointsAroundLine(a, b):
    X = []
    Y = []
    for i in range(50):
        x = random.uniform(-1, 1)
        y = a*x + b
        X.append(x + random.uniform(-0.2, 0.2))
        Y.append(y + random.uniform(-0.2, 0.2))
    return np.array(X).reshape((-1, 1)), np.array(Y)

def testPoint():
    X, Y = randomPointsAroundLine(1, 0)
    x_train = X[:20]
    x_test = X[-20:]

    y_train = Y[:20]
    y_test = Y[-20:]

    plt.scatter(x_train, y_train, color = 'blue')
    plt.scatter(x_test, y_test, color = 'black')
    plt.show()

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    print(regr.coef_, regr.intercept_)

    y_pred = regr.predict(x_test)
    print('Mean squared error: %2f' % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    
    plt.scatter(x_train, y_train, color='green') 
    plt.scatter(x_test, y_test, color='black') 
    plt.scatter(x_test, y_pred, color='blue') 
    plt.plot(x_test, y_pred, color='blue', linewidth = 3)
    plt.show()

if __name__ == '__main__':
    testPoint()