import numpy as np
from sklearn.externals import joblib

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([0, 0, 0, 1, 1, 1])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
GaussianNB(priors=None, var_smoothing=1e-09)
print(clf.predict([[3, 3]]))


joblib.dump(clf,'test.joblib')
