import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
np.random.seed(44)

epochs = 100
learnrate = 0.01

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = -(y - output)
    weights -= learnrate * d_error * x
    bias -= learnrate * d_error
    return weights, bias

def train(features, targets, epochs, learnrate):
    n_records, n_features = features.shape
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    for e in range(epochs):
        for x, y in zip(features, targets):
            #print(x,y)
            output = output_formula(x, weights, bias)
            error = error_formula(y, output)
            weights, bias = update_weights(x, y, weights, bias, learnrate)
    return weights,bias


def predict(features, targets, weights,bias):
    out = output_formula(features, weights, bias)
    predictions = out > 0.5
    accuracy = np.mean(predictions == targets)
    print("Vaidation Accuracy: ", accuracy)
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report,accuracy_score,precision_recall_fscore_support
    cm=confusion_matrix(predictions,targets)
    print(cm)
    print(classification_report(predictions,targets))
    print(accuracy_score(predictions,targets))
    print(precision_recall_fscore_support(predictions,targets))
    
def predict3(features, targets, weights,bias):
    out = output_formula(features, weights, bias)
    predictions = out > 0.7
    accuracy = np.mean(predictions == targets)
    print("Vaidation Accuracy: ", accuracy)
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report,accuracy_score,precision_recall_fscore_support
    cm=confusion_matrix(predictions,targets)
    print(cm)
    print(classification_report(predictions,targets))
    print(accuracy_score(predictions,targets))
    print(precision_recall_fscore_support(predictions,targets))
    
def predict2(features, targets, weights,bias,k):
    out = output_formula(features, weights, bias)
    predictions = out > k
    accuracy = np.mean(predictions == targets)
    #print("Vaidation Accuracy: ", accuracy)
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report,accuracy_score,precision_recall_fscore_support
    #cm=confusion_matrix(predictions,targets)
    #print(cm)
    #print(classification_report(predictions,targets))
    #print(accuracy_score(predictions,targets))
    #print(precision_recall_fscore_support(predictions,targets)) 
    p,r,f,s=precision_recall_fscore_support(predictions,targets)
    #print(p,r)
    return p,r
            
def run():
    from sklearn.model_selection import train_test_split
    dataset=pd.read_csv('AdmissionDataset/data.csv')
    X=dataset.iloc[:,1:8].as_matrix()
    Y=dataset.iloc[:,8].values
    for i in range(0,len(Y)):
        if(Y[i]>=0.5):
            Y[i]=1
        else:
            Y[i]=0
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0) 
    weights,bias=train(X_train, Y_train, epochs, learnrate)
    predict(X_test, Y_test,weights,bias)
    
def run3():
    from sklearn.model_selection import train_test_split
    dataset=pd.read_csv('AdmissionDataset/data.csv')
    X=dataset.iloc[:,1:8].as_matrix()
    Y=dataset.iloc[:,8].values
    for i in range(0,len(Y)):
        if(Y[i]>=0.7):
            Y[i]=1
        else:
            Y[i]=0
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=100) 
    weights,bias=train(X_train, Y_train, epochs, learnrate)
    predict3(X_test, Y_test,weights,bias)
    
    
def run2(k):
    from sklearn.model_selection import train_test_split
    dataset=pd.read_csv('AdmissionDataset/data.csv')
    X=dataset.iloc[:,1:8].as_matrix()
    Y=dataset.iloc[:,8].values
    for i in range(0,len(Y)):
        if(Y[i]>=k):
            Y[i]=1
        else:
            Y[i]=0
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0) 
    weights,bias=train(X_train, Y_train, epochs, learnrate)
    p,r=predict2(X_test, Y_test,weights,bias,k)
    return p,r