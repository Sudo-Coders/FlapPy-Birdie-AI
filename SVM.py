import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

data = pd.read_csv('Data.csv')

X_data_use = pd.DataFrame()
X_data_use['X'] = data['X']
X_data_use['Y'] = data['Y']
y_data_use = data['click']

# print X_data_use.shape
# print y_data_use.shape

# oversampling of data
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X_data_use, y_data_use)

# print X_resampled.shape
# print y_resampled.shape

# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

clf = svm.SVC()
clf.fit(X_resampled, y_resampled)

# y_pred = clf.predict(X_test)

# print accuracy_score(y_test, y_pred)

joblib.dump(clf, 'classifier.pkl')

print "DONE!!!!"