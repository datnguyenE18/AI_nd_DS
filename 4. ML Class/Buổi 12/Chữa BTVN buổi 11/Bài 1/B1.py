import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import csv

# Xây dựng hàm sinh ngẫu nhiên 1 tập dữ liệu::
def randomPointsAroundHyperlane( w, b, n ):
    X = []
    Y = []
    for i in range( n ):
        x = np.random.uniform( -1.0, 1.0, size = w.shape )
        y = np.dot( x, w ) + b
        
        X.append( x + np.random.uniform( -0.1, 0.1, size = x.shape ) )
        Y.append( y + np.random.uniform( -0.1, 0.1 ) )
    return np.array( X ), np.array( Y ) 

# Áp dụng hàm trên để sinh dữ liệu:
w = np.random.uniform( low=-1.0, high=1.0, size=(5) )
b = random.uniform( -1, 1 )
X_train, y_train = randomPointsAroundHyperlane( w, b, 100 )
X_test, y_test = randomPointsAroundHyperlane( w, b, 5 )

# Thực hiện học và in ra tham số mô hình:
regr = linear_model.LinearRegression()
regr.fit( X_train, y_train )
print( regr.coef_, regr.intercept_ )

# Ước lượng kết quả đầu ra của tập test và in kết quả:
y_pred = regr.predict( X_test )
print( y_test ) 
print( y_pred )

# Thực hiện tính sai số của kết quả ước lượng với kết quả chính xác:
# *Sử dụng hàm được hỗ trợ (% mean_squared_error()):
print( 'Mean squared error: %f' % mean_squared_error( y_test, y_pred ) )

# *Tự code:
mse = 0
for i in range( y_test.shape[0] ):
    mse = mse + (y_test[i] - y_pred[i]) * (y_test[i] - y_pred[i])
mse = mse / y_test.shape[0]
print( 'Mean squared error: %f' % mse )

'''
[-0.09027738  0.67716538  0.35065004  0.36643236  0.14739296] -0.27158224870511083
[-1.39595678 -0.52681333  0.22919984 -0.36119723  0.47564826]
[-1.31107841 -0.28674133  0.31153716 -0.43924615  0.43038282]
Mean squared error: 0.015952
Mean squared error: 0.015952
'''