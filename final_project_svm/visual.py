import numpy as np
import matplotlib.pyplot as plt


def data_cleaning(dataset, n):
    x_list1 = []
    x_list2 = []
    for i in range(n):
        x_list1.append(dataset['data'][i][0])
        x_list2.append(dataset['data'][i][1])
    return x_list1, x_list2

alist = []

def overview(x1, x2, y):
    plt.scatter(x1, x2, c=y, cmap=plt.cm.Spectral)

def import_dataset(x1, x2, y):
    # x1:the first x variable, x2:the second x variable, y:the y variable, all in array form
    for i in range(len(x1)):
        alist.append([x1[i] / 10, x2[i] / 10])
    x = np.array(alist)
    return x, y



def visualize_svm(X, y, clf):
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, cmap=plt.cm.Spectral)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "g--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()


def accuracy(y, predictions):
    print(f"Accuracy: {sum(y == predictions) / y.shape[0]}")
    return sum(y == predictions) / y.shape[0]

# if __name__ == "__main__":
#     # Imports
#     from sklearn import datasets
#     import matplotlib.pyplot as plt
#
#     clf = SVM()
#     clf.fit(X, y)
#
#     predictions = clf.predict(X)
