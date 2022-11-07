# CV: https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
# sklearn cv: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes
# CV hyperparameter tunning: https://dev.to/balapriya/cross-validation-and-hyperparameter-search-in-scikit-learn-a-complete-guide-5ed8
# higher terms (e.g., x1**2, x1*x2: https://realpython.com/linear-regression-in-python/)

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np


## Load the diabetes dataset
# diabetes = datasets.load_diabetes() #by deafult, it will give a Buncy object
# diabetes.keys()
#
# diabetes.feature_names
# diabetes.data.shape
# diabetes.target.shape

diabetes = datasets.load_diabetes(as_frame=True)
diabetes.data # will give pandas df
diabetes.target

# diabetes = datasets.load_diabetes(return_X_y=True, as_frame=True) # gives tuples
# len(diabetes)
# type(diabetes[0]) # same with above. it is df.
# type(diabetes[1])

x = diabetes.data
y = diabetes.target

## create training and testing vars
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

## fit a model
lm = linear_model.LinearRegression()
model = lm.fit(x_train,y_train)
predictions = lm.predict(x_test)
predictions[:5]

## visualize: The line / model
plt.scatter(y_test,predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
z = np.linspace(0,300,301)
plt.plot(z,z,'r')
plt.show()

## Evaluation
print(model.score(x_test,y_test)) # R**2: .score(X,y) Return the coefficient of determination of the prediction

## Cross Validation
'''
Purpose of CV 
1. check the average model performance using CV => cross_val_score, cross_val_predict
2. hyper parameter tunning => GridSearchCV
Crossvalidation score와 predict로는 모델(e.g., lm)의 평균적인 퍼포먼스만 알 수 있지, 최종적으로 k-CV로 나오는,
최종 모델을 찾아 낼 수 없다.
'''

# 1. Check the average model performance.
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

cross_val_score(lm,x,y,cv=5) # (model, train data, train lael, cv numbers)
cv_predictions = cross_val_predict(lm,x,y,cv=20) # show prediction results for each validation set * 20
plt.scatter(y,cv_predictions)
plt.show()

cv_predictions = cross_val_predict(lm,x,y,cv=2) # show prediction results for each validation set * 2
plt.scatter(y,cv_predictions)
plt.show()

## Evaluation
rsq = metrics.r2_score(y,cv_predictions)
print("Cross-Predicted r2: ",rsq)

# For hyper parameter tunning using GridSearch, See skl_KNN file.