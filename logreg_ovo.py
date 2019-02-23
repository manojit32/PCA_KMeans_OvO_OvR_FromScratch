import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
np.random.seed(44)

epochs = 25
learnrate = 0.004

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


def predict(features, targets, weights,bias,result):
    y=[]
    for x in features:
        o=[]
        pred=[]
        for i in range(len(bias)):
            out = output_formula(x, weights[i], bias[i])
            predictions = out > 0.5
            #o.append(predictions)
            o.append(out)
        if max(o)>=0.5:
            val=result[o.index(max(o))][0]
        else:
            val=result[o.index(max(o))][1]
        #print(o)
        for i in range(len(o)):
            if o[i]==True:
                pred.append(result[i])
        new_pred=[]        
        for [i,j] in pred:
            new_pred.append(i)
            new_pred.append(j)
        cnt=0
        #print(new_pred)
        p=0
        for i in list(set(new_pred)):
            c=new_pred.count(i)
            if (cnt <= c):
                cnt=c
                p=i
        #y.append(p)
        '''if val.count(9)==1:
            y.append(val[0])
        else:
            t=np.random.randint(0,2)
            y.append(val[t])'''
        y.append(val)
    #print(y)
    #print(targets)
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
    y=dataset.iloc[:,11]
    result = []
    source=list(set(dataset.iloc[:,11]))
    for p1 in range(len(source)):
        for p2 in range(p1+1,len(source)):
            result.append([source[p1],source[p2]])
    #print((result))
    w=[]
    b=[]
    acc=[]
    for [i,j] in result:
        dataset=pd.read_csv('wine-quality/data.csv',sep=";")
        d1=dataset
        for m in source:
            if ((m!=i) & (m!=j)): 
                d1 = d1[d1['quality""']!=m]
        
        X=d1.iloc[:,0:11].as_matrix()
        y1=d1.iloc[:,11].values
        y2=y1
        #print(y1)
        for k in range(len(y2)):
            if y2[k]==i:
                y2[k]=1
            else:
                y2[k]=0
        #print(y2)
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()
        X=scaler.fit_transform(X)
        weights,bias=train(X,y2, epochs, learnrate)
        w.append(weights)
        b.append(bias)
    #print(w)
    #print(b)
    dataset=pd.read_csv('wine-quality/data.csv',sep=";")
    X=dataset.iloc[:,0:11].as_matrix()
    y=dataset.iloc[:,11].values
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    a=predict(x_test,y_test,w,b,result)
    
    
    
    
    
    
    
    
    
    