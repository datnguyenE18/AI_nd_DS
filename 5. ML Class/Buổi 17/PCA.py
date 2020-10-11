import numpy as np
from sklearn import decomposition

from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

class PCA:
    def __init__(self):
        self.eigenVals = None
        self.eigenVecs = None
        self.mean = None

    def calc(self, X):
        n, m = X.shape

        self.mean = np.mean(X, 0)

        C = np.dot((X-self.mean).T, (X-self.mean)) / (n-1)

        self.eigenVals, self.eigenVecs= np.linalg.eig(C)

    def project(self, X):
        return np.dot((X-self.mean), self.eigenVecs)

    def backproject(self, X):
        return np.dot(X, self.eigenVecs.T) + self.mean

def testPCA1():
    X = np.random.randint(0, 6, (4, 3))
    print("origin", X)

    pca1 = PCA()
    pca1.calc(X)

    print("eigenvalues", pca1.eigenVals)
    print("eigenvectors", pca1.eigenVecs)
    p = pca1.project(X)

    print("projection", p)
    print("restore", pca1.backproject(p))

def testPCA2():
    X = np.random.randint(0, 6, (4, 3))
    print("origin", X)

    pca2 = decomposition.PCA()
    pca2.fit(X)
    print("eigenvectors", pca2.components_)

    p1 = pca2.transform(X)

    print("projection", p1)
    print("restore", pca2.inverse_transform(p1))


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def testFaces():
    # Tải bộ dữ liệu LFW được dựng sẵn
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # lấy các kích thước của tập ảnh dữ liệu
    n_samples, h, w = lfw_people.images.shape
    print(n_samples, h, w)

    # lấy đặc trưng ảnh
    X = lfw_people.data
    n_features = X.shape[1]

    # lấy các nhãn
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    # in một số thông tin
    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    # chia tập học và tập test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)

    # tính PCA trên tập dữ liệu học, chọn 150 thành phần
    n_components = 150
    pca = decomposition.PCA(n_components=n_components).fit(X_train)

    # lấy các eigenface
    eigenfaces = pca.components_.reshape((n_components, h, w))

    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    plt.show()

    # tính các đặc trưng PCA
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # xây dựng bộ phân lớp và học
    clf = LogisticRegression()
    clf.fit(X_train_pca, y_train)

    # Tính nhãn ước lượng
    y_pred = clf.predict(X_test_pca)

    # in các tham số
    print("confusion_matrix", confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print("classification_report", classification_report(y_test, y_pred, target_names=target_names))



    prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

    plot_gallery(X_test, prediction_titles, h, w)

    plt.show()


def test():
    testPCA1()
    #testPCA2()
    #testFaces()