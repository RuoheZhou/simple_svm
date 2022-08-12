import numpy as np
from visual import import_dataset, visualize_svm, accuracy, data_cleaning, overview

class svm():
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lrate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lrate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lrate * (
                            2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lrate * y_[idx]

    def predict(self, X):
        approx1 = []
        approx = np.dot(X, self.w) - self.b
        for i in approx:
            if i <= 0:
                a = 0
            else:
                a = 1
            approx1.append(a)
        return approx1

if __name__ == "__main__":

    #n means the number of data points you want to include in SVM
    n = 200

    #import data
    from sklearn import datasets
    cancer = datasets.load_breast_cancer()
    x_list1, x_list2 = data_cleaning(cancer, n)
    X, y = import_dataset(x_list1, x_list2, cancer['target'][0:n])
    print(type(X))
    # an overview of the distribution of the data
    overview(x_list1, x_list2, cancer['target'][0:n])

    clf = svm()
    clf.fit(X, y)

    #calculate the accuracy of prediction
    predictions = clf.predict(X)
    accuracy(y, predictions)
    print(type(accuracy(y, predictions)))
    #visualize the hyperplane
    visualize_svm(X, y, clf)