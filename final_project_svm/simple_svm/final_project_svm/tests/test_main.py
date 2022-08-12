from main import svm
from visual import import_dataset, accuracy, data_cleaning, overview

import numpy as np
import unittest

n = 200

# import data
from sklearn import datasets

cancer = datasets.load_breast_cancer()
x_list1, x_list2 = data_cleaning(cancer, n)
X, y = import_dataset(x_list1, x_list2, cancer['target'][0:n])
# an overview of the distribution of the data
overview(x_list1, x_list2, cancer['target'][0:n])

clf = svm()
clf.fit(X, y)

# calculate the accuracy of prediction
predictions = clf.predict(X)
accuracy(y, predictions)

def test_x():
    return X

def test_y():
    return y

def test_predict():
    return accuracy(y, predictions)

class TestLList(unittest.TestCase):
    def test_dataset1(self):
        k = test_x()
        data_type = np.ndarray
        self.assertEqual(type(k), data_type)
    def test_dataset2(self):
        k = test_y()
        data_type = np.ndarray
        self.assertEqual(type(k), data_type)
    def test_accuracy(self):
        k = test_predict()
        data_type = np.float64
        self.assertEqual(type(k), data_type)