import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
np.random.seed(44)

epochs = 50
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
    y=[]
    for x in features:
        o=[]
        for i in range(len(bias)):
            out=output_formula(x,weights[i],bias[i])
            o.append(out)
        y.append(o.index(max(o))+3)
    #print(y)
    #print(targets)
    '''out = output_formula(features, weights, bias)
    predictions = out > 0.5'''
    accuracy = np.mean(y == targets)
    print("Vaidation Accuracy: ", accuracy)
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report,accuracy_score,precision_recall_fscore_support
    cm=confusion_matrix(y,targets)
    print(cm)
    print(classification_report(y,targets))
    print(accuracy_score(y,targets))
    #print(precision_recall_fscore_support(predictions,targets))
    return accuracy_score(y,targets)
            
def run():
    from sklearn.model_selection import train_test_split
    dataset=pd.read_csv('wine-quality/data.csv',sep=";")
    X=dataset.iloc[:,0:11].as_matrix()
    y=dataset.iloc[:,11].values
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    #print(y)
    acc=[]
    w=[]
    b=[]
    for i in list(set(y)):
        dataset=pd.read_csv('wine-quality/data.csv',sep=";")
        y1=dataset.iloc[:,11].values
        y2=y1
        for j in range(len(y2)):
            if y2[j]==i:
                y2[j]=1
            else:
                y2[j]=0 
        weights,bias=train(X,y2, epochs, learnrate)
        w.append(weights)
        b.append(bias)
    dataset=pd.read_csv('wine-quality/data.csv',sep=";")
    X=dataset.iloc[:,0:11].as_matrix()
    y=dataset.iloc[:,11].values
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    a=predict(X,y,w,b)
        

        
    
    
    
    
    
    
    
    
    