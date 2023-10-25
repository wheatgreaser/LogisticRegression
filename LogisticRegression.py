from sklearn import linear_model
import pandas
import numpy

X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)

y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logreg = linear_model.LogisticRegression()
logreg.fit(X,y)
print(logreg.predict(numpy.array([3.8]).reshape(-1, 1)))