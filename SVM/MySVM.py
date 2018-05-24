import numpy as np
import pylab as pl
from sklearn import svm

##随机创建40个点
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

##训练模型
clf = svm.SVC(kernel = 'linear')
clf.fit(X, Y)

##得到超平面线
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 50)
yy = a * xx - (clf.intercept_[0] / w[1])

print("w = ", w)
print("a = ", a)

##绘制图像
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy, 'k--')
pl.plot(xx, yy, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 80, facecolors = 'none')

for i in X:
    res=clf.predict(np.array(i).reshape(1, -1))
    if res > 0:
        pl.scatter(i[0],i[1],c='r',marker='.')
    else :
        pl.scatter(i[0],i[1],c='g',marker='.')

pl.axis('tight')
pl.show()









