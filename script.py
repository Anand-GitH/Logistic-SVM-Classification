import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    #Adding the bias term to the data
    train_data=np.hstack((np.ones((n_data,1)),train_data))
    
    #class probablity using softmax function
    probc=sigmoid(np.dot(train_data,initialWeights)).reshape(-1,1)
    classprob=np.hstack(((1-probc),probc))
    
    #Finding Error 
    y=labeli
    y=np.hstack((1-y,y))
    result=np.multiply(y,np.log(classprob))
    error= -1*np.mean(np.sum(result,axis=1))
    
    #Finding Error Gradient for the weights
    thetay= classprob[:,1]-y[:,1]
    thetay=thetay.reshape(1,-1)
    error_grad= (1/n_data)*(thetay @ train_data)
    
    return error, error_grad.flatten()

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    ypred = np.zeros((data.shape[0],W.shape[1]))
    label = np.zeros((data.shape[0], 1))

    data_bias= np.hstack((np.ones((data.shape[0],1)),data))
    classprob= sigmoid(np.dot(data_bias,W))
    label=np.argmax(classprob,axis=1).reshape(-1,1)

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    initialWeights_b=params.reshape(n_feature+1,n_class)
       
    #Adding the bias term to the data and initialization
    train_data=np.hstack((np.ones((n_data,1)),train_data))
    y=labeli
    
    #Calculating probability of the class for each data point and normalizing
    smax=np.exp(np.dot(train_data,initialWeights_b))
    norms=np.sum(smax,axis=1).reshape(-1,1)
    probc=smax/norms
    
    #Finding Error
    error=-1*np.sum(np.multiply(y,np.log(probc)))
    
    for i in range(n_class):
        p_y=probc[:,i]-y[:,i]
        p_y=p_y.reshape(1,-1)
        error_grad[:,i]= np.dot(p_y,train_data)
    
    return error, error_grad.flatten()

def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    ypred = np.zeros((data.shape[0],W.shape[1]))
    label = np.zeros((data.shape[0], 1))
    
    data_bias = np.hstack((np.ones((data.shape[0],1)),data))
    
    smax=np.exp(np.dot(data_bias,W))
    norms=np.sum(smax,axis=1).reshape(-1,1)
    classprob=smax/norms
    
    label=np.argmax(classprob,axis=1).reshape(-1,1)

    return label

########################################################Logistic Regression################################################
"""
Script for Logistic Regression
"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))

for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()


#Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))

opts = {'maxiter': 100}


for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

print("Logistic Regression - Mutliple binary classifiers combined to build multi class classifier")
print("\n########################################################################################################################\n")
#Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

#Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

#Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

print("\n########################################################################################################################\n")
########################################################################################################################

"""
Script for Extra Credit Part
"""

##################################################Logistic Multiclass classifier######################################

# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

print("Logistic Regression - Mutliclass Classifier")

print("\n########################################################################################################################\n")
# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

print("\n########################################################################################################################\n")


#########################################################################################################################

"""
#Script for Support Vector Machine
"""

print('\n\n--------------Support Vector Machines-------------------\n\n')
#Choosing randomly 10000 samples of training
indx=np.random.choice(n_train,10000, replace=False)
sample_data=train_data[indx,:]
sample_label=train_label[indx,:].reshape(-1)

#1. Linear Kernel
print('##############Linear Kerner#####################')
linear_svc = svm.SVC(kernel='linear')
linear_svc.fit(sample_data,sample_label)

# Find the accuracy on Training Dataset
predicted_label = linear_svc.predict(train_data).reshape(-1,1)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = linear_svc.predict(validation_data).reshape(-1,1)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = linear_svc.predict(test_data).reshape(-1,1)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


#radial basis function and gamma=1
print('##############Radial Basis with Gamma=1#####################')
rbf_svc = svm.SVC(kernel='rbf',gamma=1)

rbf_svc.fit(sample_data,sample_label)

# Find the accuracy on Training Dataset
predicted_label = rbf_svc.predict(train_data).reshape(-1,1)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = rbf_svc.predict(validation_data).reshape(-1,1)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = rbf_svc.predict(test_data).reshape(-1,1)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


#radial basis function
print('##############Radial Basis with Gamma default#####################')
rbf_svc = svm.SVC(kernel='rbf')

rbf_svc.fit(sample_data,sample_label)
# Find the accuracy on Training Dataset
predicted_label = rbf_svc.predict(train_data).reshape(-1,1)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = rbf_svc.predict(validation_data).reshape(-1,1)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = rbf_svc.predict(test_data).reshape(-1,1)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

#radial basis function with c C (1; 10; 20; 30; ; 100)
train_acc=[]
valid_acc=[]
test_acc=[]

print('##############Radial Basis with Gamma default and different values of C#####################')
cval=[1,10,20,30,40,50,60,70,80,90,100]
for each in (cval):
    
    rbf_svc = svm.SVC(kernel='rbf',C=each)

    rbf_svc.fit(sample_data,sample_label)

    # Find the accuracy on Training Dataset
    predicted_label = rbf_svc.predict(train_data).reshape(-1,1)
    train_acc.append(100 * np.mean((predicted_label == train_label).astype(float)))

    # Find the accuracy on Validation Dataset
    predicted_label = rbf_svc.predict(validation_data).reshape(-1,1)
    valid_acc.append(100 * np.mean((predicted_label == validation_label).astype(float)))

    # Find the accuracy on Testing Dataset
    predicted_label = rbf_svc.predict(test_data).reshape(-1,1)
    test_acc.append(100 * np.mean((predicted_label == test_label).astype(float)))



df=pd.DataFrame({'C': cval, 'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc })
 
# multiple line plot
plt.plot( 'C', 'train_acc', data=df, marker='', color='#ff7f0e', linewidth=2, label="Training Accuracy")
plt.plot( 'C', 'valid_acc', data=df, marker='', color='#d62728', linewidth=2, label="Validation Accuracy")
plt.plot( 'C', 'test_acc', data=df, marker='', color='#bcbd22', linewidth=2,  label="Test Accuracy")
plt.title("Choosing cost parameter C for SVM")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend()

#After learning parameter
print('##############Final after learning parameters#####################')
rbf_svc = svm.SVC(kernel='rbf',C=10)


rbf_svc.fit(sample_data,sample_label)

# Find the accuracy on Training Dataset
predicted_label = rbf_svc.predict(train_data).reshape(-1,1)

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = rbf_svc.predict(validation_data).reshape(-1,1)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = rbf_svc.predict(test_data).reshape(-1,1)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
