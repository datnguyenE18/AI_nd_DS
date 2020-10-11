import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random

def testDiabetes():
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    x_train = diabetes_X[:-20]
    x_test = diabetes_X[-20:]

    y_train = diabetes_y[:-20]
    y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)
    print(regr.coef_, regr.intercept_)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The mean squared error:
    print('Mean squared error: %2f' % mean_squared_error(y_test, y_pred))

    # The coefficient of determination: 1 is perfect prediction:
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

    # Plot outputs:
    #plt.scatter(x_train, y_train, color = 'green') 
    plt.scatter(x_test, y_test, color = 'black') 
    plt.scatter(x_test, y_pred, color = 'blue') 
    plt.plot(x_test, y_pred, color = 'blue', linewidth = 2)

    plt.show()

if __name__ == "__main__":
    testDiabetes()
