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

train_data = np.loadtxt(open("debugdevelopmentTrainingSet.csv"),
    delimiter = ",", 
    skiprows=0,
    dtype=np.float64
    )
X_train = train_data[:,0:10]    

'''
#Turns out I didn't need to change my data format once I realized I couldn't do SVM on 
#floating point.  Switched to regression and it worked. 
#Y_train needs to be an array with one list.  This is the only way I know to do it.
Y_train_1 = train_data[:,10:11]       #This gives an array of num_samples lists of 1.
import itertools
Y_train = np.array(list(itertools.chain(*Y_train_1))) #This merges the many lists to 1 then creates np.array
print Y_train
'''

###############################################################################
# Compute a PCA                                                               #
###############################################################################
from sklearn.decomposition import RandomizedPCA
n_components = 2

pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
#print pca.explained_variance_ratio_

X_train_pca = pca.transform(X_train)
#print type(X_train)
print type(X_train_pca)
print type(Y_train)

###############################################################################
# Compute a Regression                                                        #
###############################################################################
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit (X_train_pca,Y_train_1) 
reg.coef_



#### !!!! Can't use rational values for SVM/decision trees or other classifiers.  Need regression !!!! ####

'''
###############################################################################
# Train a SVM classification model                                            #
###############################################################################

from sklearn.svm import SVC

print "Fitting the classifier to the training set"

clf = SVC()
clf.fit(X_train_pca, Y_train)  
'''

