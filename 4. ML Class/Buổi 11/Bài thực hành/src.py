import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random
import csv

def testDiabetes():
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)

    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(diabetes_y_test, diabetes_y_pred))

    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

def randomPointsAroundLine(a, b):
    X = []
    Y = []
    for i in range(50):
        x = random.uniform(-1, 1)
        y = a*x+b
        X.append(x + random.uniform(-0.2, 0.2))
        Y.append(y + random.uniform(-0.2, 0.2))
    return np.array(X).reshape((-1, 1)), np.array(Y)

def testPoints():
    X, Y = randomPointsAroundLine(1, 0)

    # Split the targets into training/testing sets
    X_train = X[:-20]
    X_test = X[-20:]

    y_train = Y[:-20]
    y_test = Y[-20:]

    plt.scatter(X_train, y_train, color='green')
    plt.scatter(X_test, y_test, color='black')

    plt.xticks(())
    plt.yticks(())

    plt.show()

    regr = linear_model.LinearRegression()

    regr.fit(X_train, y_train)

    # f(x1,..,xn) = a0 + a1*x1 + a2*x2 + … +  an*xn
    # coef_ là [a1, a2,..,an]
    # intercept_ là a0
    print(regr.coef_, regr.intercept_)
    y_pred = regr.predict(X_test)

    #plt.scatter(X_train, y_train, color='green')
    plt.scatter(X_test, y_test, color='black')
    plt.scatter(X_test, y_pred, color='blue')

    plt.xticks(())
    plt.yticks(())

    plt.show()

if __name__ == "__main__":
    testPoints()
    testDiabetes()
