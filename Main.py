import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

print(cancer_data.DESCR)

print(cancer_data.data.shape)
print(cancer_data.target.shape)

print("First patient in database")
print(cancer_data['data'][0,:])


train_data, test_data, \
train_target, test_target = \
train_test_split(cancer_data.data, cancer_data.target, test_size=0.1)

logistic_regression = LogisticRegression()
logistic_regression.fit(train_data, train_target)

conf_matrix = confusion_matrix(test_target, logistic_regression.predict(test_data))
print("Confusion_matrix:")
print(conf_matrix)

from sklearn.metrics import accuracy_score
acc = accuracy_score(test_target, logistic_regression.predict(test_data))
print("Model accuracy is {0:0.2f}".format(acc))

import seaborn as sns;
sns.heatmap(conf_matrix)