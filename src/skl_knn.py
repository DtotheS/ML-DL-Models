# reference: https://dev.to/balapriya/cross-validation-and-hyperparameter-search-in-scikit-learn-a-complete-guide-5ed8
# KNN: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761#:~:text=KNN%20works%20by%20finding%20the,in%20the%20case%20of%20regression).
# K-NN: test[i]에서 가까운순으로 k개의 labeled data 뽑는다 => majority voting으로 가장 많은 label을 받은 것을 test[i]의 label로 결정한다. (tie rule은 여러개 있음. e.g., 전체 총 거리 따지던지..)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib import pyplot as plt

iris = load_iris(as_frame=True)

x = iris.data
y = iris.target

xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state=4,test_size=0.2) # Let’s create the train and test sets with random_state= 4. Setting the random_state ensures reproducibility.

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(xtrain,ytrain)
ypred = knn.predict(xtest)
metrics.accuracy_score(ytest,ypred)

plt.scatter(ytest,ypred) # cause there are 4 neighbors
plt.show()


