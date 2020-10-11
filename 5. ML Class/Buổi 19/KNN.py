from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier


def test():
    digits = load_digits()

    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

    nNeighbors = 7
    clf = KNeighborsClassifier(n_neighbors=nNeighbors)
    clf.fit(X_train, y_train)

    dists, inds = clf.kneighbors(X_test)
    results = clf.predict(X_test)

    for si in range(10):
        print("pred:", results[si], ", truth:", y_test[si], "------------------")
        for i in range(nNeighbors):
            print("index:", inds[si][i], ", label :", y_train[inds[si][i]], ", distance :", dists[si][i])
