import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

# Global Functions
def mystats(truth, pred, text=None):
    '''
        Scoring Calculations ported from matlab script
        
        :params
            truth: y column with expected value
            pred: predicted value of y by model
            
        Returns tuple or string (if text==True)
    '''
    def mad(arr):
        """ Median Absolute Deviation: a "Robust" version of standard deviation.
            Indices variabililty of the sample.
            https://en.wikipedia.org/wiki/Median_absolute_deviation 
        """
        arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
        med = np.median(arr)
        return np.median(np.abs(arr - med))

    error = pred - truth
    cr5 = len(error[(error < 5) & (error > -5)]) * 1.0 / len(error)
    
    mean_error = np.mean(error)
    std_error = np.std(error)
    mad_error = mad(error)
    if text:
        return 'MeanError= {0:0.3f}, StdError= {1:0.3f}, MadError={2:0.3f}, CR5= {3:0.3f}'.format(mean_error, std_error, mad_error, cr5)
    return (mean_error, std_error, mad_error, cr5)


def myplot2(y, yt, pred_train, pred, title):
    '''
    :Params
    y: Training data set truth column  (11 in our example)
    yt: Testing data set truth column 
    pred_train: estimated predictions based on model for training dataset
    pred: estimated predictions for testing dataset
    
    :Return
    2x3 plot of Truth vs Predicted, Truth vs error, Predicted vs error for the training and testing datasets
    '''
    fig , ax = plt.subplots(2, 3)
    error = pred - yt
    error_train = pred_train - y
    
    # Train Dataset
    c1 = sns.color_palette()[2]
    sns.regplot(y, pred_train, ax=ax[0,0], color = c1)
    sns.regplot(y, error_train, ax=ax[0,1], color = c1)
    sns.regplot(pred_train, error_train, ax=ax[0,2], color = c1)
    
    # Test Dataset
    c2 = sns.color_palette()[1]
    sns.regplot(yt, pred, ax=ax[1,0], color=c2)
    sns.regplot(yt, error, ax=ax[1,1], color=c2)
    sns.regplot(pred, error, ax=ax[1,2], color=c2)
    
    ax[1,0].axes.set_xlabel('Truth')
    ax[1,0].axes.set_ylabel('Test \n Predicted')
    ax[1,0].axes.set_ylim([-40,20])
    ax[1,0].axes.set_xlim([-40,20])
    ax[1,1].axes.set_xlabel('Truth')
    ax[1,1].axes.set_ylabel('Error')
    ax[1,2].axes.set_xlabel('Predicted')
    ax[1,2].axes.set_ylabel('Error')
    
    ax[0,0].axes.set_xlabel('Truth')
    ax[0,0].axes.set_ylabel('Train \n Predicted')
    ax[0,0].axes.set_ylim([-40,20])
    ax[0,0].axes.set_xlim([-40,20])
    ax[0,1].axes.set_xlabel('Truth')
    ax[0,1].axes.set_ylabel('Error')
    ax[0,2].axes.set_xlabel('Predicted')
    ax[0,2].axes.set_ylabel('Error')
    
    fig.set_figwidth(12)
    fig.set_figheight(8)

    
    plt.suptitle('{0} \n {1}'.format(title, mystats(yt, pred, text=1)))
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
def plot_X_error(X, y, pred):
    error = pred-y
    error.name = 'error'
    fig , ax = plt.subplots(2, 5, sharey=True)
    for ind, feat in enumerate(features):
        yind = 0 if ind < 5 else 1
        ind = ind if ind < 5 else ind - 5
        sns.regplot(X[feat], error, ax=ax[yind, ind])
    fig.set_figwidth(20)
    fig.set_figheight(8)
    fig.suptitle('Feature vs Error', fontsize=20)

##########################################################
# Import and separate the test data and the test results #
##########################################################

#get training data
train_data = np.loadtxt(open("developmentTrainingSet.csv"),
    delimiter = ",", 
    skiprows=0,
    dtype=np.float64
    )
X_train = train_data[:,0:10]    
y_train = train_data[:,10:11] 
y_train = y_train.ravel()   #ravel takes a column vector and changes it to a 1D array

#get testing data
test_data = np.loadtxt(open("developmentTestingSet.csv"),
    delimiter = ",", 
    skiprows=0,
    dtype=np.float64
    )
X_test = test_data[:,0:10]    
y_test = test_data[:,10:11] 
y_test = y_test.ravel()   #ravel takes a column vector and changes it to a 1D array
'''
#get evaluation data
test_data = np.loadtxt(open("developmentValidationSet.csv"),
    delimiter = ",", 
    skiprows=0,
    dtype=np.float64
    )
X_test = test_data[:,0:10]    
'''

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

#print (X_test)
#print type(X_train_pca)
#print (y_test)
'''

###############################################################################
# Compute a Neural Network                                                    #
###############################################################################
# MSE = 73.21, Variance score = 0.0158

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

X_test_pca = X_test
X_train_pca = X_train

parameters = {'hidden_layer_sizes':[1,5], 'solver':('lbfgs','adam'), \
    'max_iter':[10,100],'alpha':[1,1000],'activation':('logistic','relu')}

mlpr = MLPRegressor()

clf = GridSearchCV(mlpr,parameters)
clf.fit (X_train_pca,y_train) 

#print(clf.best_params_)
#print (clf.get_params())
# The coefficients
#print('Coefficients: \n', clf.coefs_)

myerr = clf.predict(X_test_pca) - y_test

# The mean squared error
print("Mean squared error: %.3f" % np.mean(myerr ** 2))

# The mean error
print("MeanError: %.3f" % np.mean(myerr))

# The std error (StdError)
print("StdError: %.3f" % np.std(myerr))

# The median absolute error
med = np.median(myerr)
mad = np.median(np.abs(myerr - med))
print("MadError: %.3f"
      %mad )
      
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % clf.score(X_test_pca, y_test))

#print(clf.predict(X_test_pca))

pred_train = clf.predict(X_train_pca)
pred_test = clf.predict(X_test_pca)


###############################################################################
# Plot Data                                                                   #
###############################################################################


#mystats(y_test, pred_test, text=1)
myplot2(y_train, y_test, pred_train, pred_test, 'NLPRegressor')

# Plot outputs
#plt.scatter(clf.predict(X_test_pca), y_test,  color='black')
plt.show()

#np.savetxt("validation.csv", clf.predict(X_test_pca))
#np.savetxt("y_test.csv", y_test)

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