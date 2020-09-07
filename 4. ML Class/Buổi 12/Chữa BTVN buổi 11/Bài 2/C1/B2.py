import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import csv

def readForestFiresCsv():
    X = []
    Y = []
    with open( 'forestfires.csv', mode='r' ) as csv_file:
        csv_reader = csv.DictReader( csv_file )
        for row in csv_reader:
            Y.append( float( row["area"] ) )
            X.append( [float( row["X"] ),float( row["Y"] ), 
                       float( row["FFMC"] ),float( row["DMC"] ),float( row["DC"] ),float( row["ISI"] ), 
                       float( row["temp"] ),float( row["RH"] ),float( row["wind"] ),float( row["rain"] )] )
    return np.array( X ), np.array( Y )

X, Y = readForestFiresCsv()

x_train = X[:-20]
x_test = X[-20:]

y_train = Y[:-20]
y_test = Y[-20:]

regr = linear_model.LinearRegression()

regr.fit( x_train, y_train )
print( "coef_ =", regr.coef_, "\n" )
print( "intercept_ =", regr.intercept_, "\n" )
y_pred = regr.predict( x_test )
print( 'Mean squared error: %f' % mean_squared_error( y_test, y_pred ) )

"""
coef_ = [ 1.97720473e+00  5.08312968e-01 -3.09645621e-02  7.91899961e-02
 -3.49526205e-03 -7.72553317e-01  8.66503255e-01 -2.13017326e-01
  1.65428553e+00 -2.09569883e+01]

intercept_ = -8.863779530240935

Mean squared error: 1038.882196
"""