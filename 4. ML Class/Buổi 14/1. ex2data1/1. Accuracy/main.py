from Buoi13_LogisticRegression import read_ex2data1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

X, Y = read_ex2data1()
x_train, x_test, y_train, y_test = train_test_split(X, Y)

regr = LogisticRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

# accuracy
corr = 0
for i in range(y_test.shape[0]):
    if y_pred[i]==y_test[i]:
        corr = corr+1
accuracy = 1.0*corr/ y_test.shape[0]
print("accuracy: ", accuracy)

# Sử dụng hàm hỗ trợ:
print("accuracy1: ", accuracy_score(y_test, y_pred))