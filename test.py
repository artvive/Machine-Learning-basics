# %% Linear Regression
from LinearRegression import LinearRegression
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

temp = np.ones((2, 4))

N = 100

X1 = np.linspace(0, 100, num=N)
X2 = np.logspace(-1, 2, num=N)
X1 = X1[:, np.newaxis]
X2 = X2[:, np.newaxis]

X = np.concatenate([X1, X2], axis=1)
noise = st.norm.rvs(0, 1, size=N)
noise = noise[:, np.newaxis]

y = 86 * X1 - 7 * X2 + noise
y = y.reshape(N)

clf = LinearRegression()

clf.fit(X, y)


y_pred = clf.predict(X)

print(clf.coef)
print(y_pred[1:10])
print(y[1:10])

# %% Logistic Regression
from LogisticRegression import LogisticRegression
N = 100

X = np.linspace(0,100, num=N)
y = np.concatenate([np.ones(N//2), np.zeros(N//2)])

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

clf = LogisticRegression()

clf.fit(X, y)

#mat = []
#for i in np.linspace(-0.1,0.1):
#    temp = []
#    for j in np.linspace(-15.5,5):
#        clf.coef_[0,0] = i
#        clf.coef_[1,0] = j
#        temp+=[clf.score(X,y)]
#    mat += [temp]
#
#mat = np.array(mat)
#
#plt.matshow(mat)
#plt.colorbar()
