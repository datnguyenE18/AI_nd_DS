import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("forestfires.csv")
data.plot(style = 'o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

X = data['X'].values.reshape(-1, 1)
Y = data['Y'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
regr = LinearRegression()
regr.fit(x_train, y_train)
print('intercept_ = ', regr.intercept_)
print('coef_ = ', regr.coef_)
y_pred = regr.predict(x_test)
print('MSE = %2f'% mean_squared_error(y_test, y_pred))
plt.scatter(x_test, y_test, color = 'green')
plt.plot(x_test, y_pred, color = 'red', linewidth = 2)
plt.xticks()
plt.yticks()
plt.show()
