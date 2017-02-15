########################
#Read in training data #
########################
#import csv  #Use to import csv files
#myfile = open('developmentTrainingSet.csv')
#myreader = csv.reader(myfile, skipinitialspace = True) #skipinitialspace makes sure it imports number only.  Otherwise can import as a string.		
#mydata = list(myreader)
#raw_data = np.array(mydata)
#X_train = raw_data[:,0:10]#.tolist()
#X = raw_data[:,10:11].tolist()

##########################################################
# Import and separate the test data and the test results #
##########################################################
import numpy as np

#get training data
train_data = np.loadtxt(open("developmentTrainingSet.csv"),
    delimiter = ",", 
    skiprows=0,
    dtype=np.float64
    )
X_train = train_data[:,0:10]    
y_train = train_data[:,10:11] 

#get testing data
train_data = np.loadtxt(open("developmentTestingSet.csv"),
    delimiter = ",", 
    skiprows=0,
    dtype=np.float64
    )
X_test = train_data[:,0:10]    
y_test = train_data[:,10:11] 

'''
#Turns out I didn't need to change my data format once I realized I couldn't do SVM on 
#floating point.  Switched to regression and it worked. 
#y_train needs to be an array with one list.  This is the only way I know to do it.
y_train_1 = train_data[:,10:11]       #This gives an array of num_samples lists of 1.
import itertools
y_train = np.array(list(itertools.chain(*y_train_1))) #This merges the many lists to 1 then creates np.array
print y_train
'''

###############################################################################
# Compute a PCA                                                               #
###############################################################################
from sklearn.decomposition import PCA
n_components = 10

#Find training PCA
pca_train = PCA(n_components=n_components).fit(X_train)
X_train_pca = pca_train.transform(X_train)

#Find testing PCA
#pca_test = RandomizedPCA(n_components=n_components, whiten=True).fit(X_test) #!!! WRONG!!! don't refit testing data.  Use training fit.
X_test_pca = pca_train.transform(X_test)

#print type(X_train)
#print type(X_train_pca)
#print type(y_train)
'''
###############################################################################
# Compute a Neural Network                                                    #
###############################################################################
# MSE = 73.21, Variance score = 0.0158
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor

clf = MLPRegressor()

clf.fit (X_train_pca,y_train) 
clf.coef_

# The coefficients
print('Coefficients: \n', clf.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((clf.predict(X_test_pca) - y_test) ** 2))
   
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % clf.score(X_test_pca, y_test))
'''


###############################################################################
# Compute a Linear Regression                                                 #
###############################################################################
# MSE = 73.21, Variance score = 0.0158
import matplotlib.pyplot as plt
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit (X_train_pca,y_train) 
regr.coef_

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test_pca) - y_test) ** 2))
   
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % regr.score(X_test_pca, y_test))

'''
# Plot outputs
plt.scatter(X_test_pca, y_test,  color='black')
plt.plot(X_test_pca, regr.predict(X_test_pca), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()
'''
'''
###############################################################################
# Compute a Ridge Regression                                                  #
###############################################################################
# MSE = 69.55, Variance score = 0.0648

import matplotlib.pyplot as plt
from sklearn import linear_model, grid_search

regr = linear_model.RidgeCV(alphas = [0.1,1.0,10.0,100.0])
regr.fit (X_train_pca,y_train) 
regr.coef_

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test_pca) - y_test) ** 2))
   
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % regr.score(X_test_pca, y_test))
'''

'''
###############################################################################
# Compute a Lasso and ElasticNet Regression                                   #
###############################################################################

import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet

#Lasso
#MSE = 78.33, Variance score = 0.065394
alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train_pca, y_train).predict(X_test_pca)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((lasso.predict(X_test_pca) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % lasso.score(X_test_pca, y_test))

#Elastic
#MSE = 78.40, Variance score = 0.065344
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train_pca, y_train).predict(X_test_pca)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((enet.predict(X_test_pca) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % enet.score(X_test_pca, y_test))
'''