import csv
import random
import math

from collections import Counter
from operator import itemgetter
from matplotlib import pyplot as plt


def csv_to_point_label(csv_file, predictor_cols, label_col):
    data = []
    with open(csv_file,'r') as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            #row=row[0].split(",")
            if (row[1])!="GRE Score":
                predictors = [float(row[i]) for i in predictor_cols]
                label = 1 if float(row[label_col])>0.7 else 0
                data.append((predictors,label))
    #print(data[0])
    return data


def split_train_test(data, ratio=0.67):
    train = []
    test = []
    for _, data_ls in enumerate(data):
        if random.random() < ratio:
            train.append(data_ls)
        else:
            test.append(data_ls)
    return train, test


def square_rooted(x):
        return round(math.sqrt(sum([a*a for a in x])),3)

def nth_root(self,value, n_root):
        root_value = 1/float(n_root)
        return round (Decimal(value) ** Decimal(root_value),3)

def euclid_dist(x,y,m):
    if m=="euclid":
        return math.sqrt(sum([(x1-x2)**2 for x1,x2 in zip(x,y)]))
    elif m=="manhat":
        return sum(abs(a-b) for a,b in zip(x,y))
    elif m=="cosine":
        numerator = sum(a*b for a,b in zip(x,y))
        denominator = square_rooted(x)*square_rooted(y)
        return round(numerator/float(denominator),3)


def get_neighbors(training_set, labeled_point, k,m):
    def point_distance(training_point):    
        return euclid_dist(training_point[0], labeled_point[0],m)
    neighbors = sorted(training_set, key=point_distance)
    k_nearest_labeled_points = neighbors[0:k]
    return k_nearest_labeled_points



def get_majority_label(labeled_points):
    labels = [label for _,label in labeled_points]
    winning_point_labels = Counter(labels).most_common()
    if len(winning_point_labels) > 1:
        winning_label = random.choice(winning_point_labels)[0]
    else:
        winning_label = winning_point_labels[0][0]
    return winning_label



def get_optimum_k(training_data, test_data,m, k_values=list(range(1,21))):
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    k_error_rates = []
    y_pred=[]
    for k in k_values:
        error_rate = 0
        pred=[]
        for labeled_point in test_data:
            neighbors = get_neighbors(training_data, labeled_point, k,m)
            predicted_label =  get_majority_label(neighbors)  
            pred.append(predicted_label)
            if predicted_label != labeled_point[1]:
                error_rate += 1/float(len(test_data))
        #print(pred)
        y_pred.append(pred)
        k_error_rates.append((k, error_rate))
    optimum_k, min_error  = sorted(k_error_rates,key=itemgetter(1))[0]
    #print(y_pred)
    y_act=[]
    for val in test_data:
        y_act.append(val[1])
    #print(y_act)
    #print(y_pred[optimum_k-2])
    #print(k_error_rates)
    print(classification_report(y_pred[optimum_k-1],y_act))
    return optimum_k, min_error, k_error_rates
            
def run(m):   
    data = csv_to_point_label("AdmissionDataset/data.csv", [1,2,3,4,5,6,7], 8)
    train_set, test_set = split_train_test(data, 0.80)
    #train_set=data[1:int(len(data)*0.8)]
    #test_set=data[int(len(data)*0.8)+1:len(data)]
    optimal_k, min_error, k_error_rates = get_optimum_k(train_set, test_set,m,
                                             k_values=list(range(2,20)))
    print("Using own KNN Algorithm ::")
    print('For k=',optimal_k,'Validation Accurcy = ',
          round(1-min_error,3))
    k_acc=[(i[0],1-i[1]) for i in k_error_rates]
    plt.plot(*zip(*k_acc), linestyle='--',marker='o', 
             markersize=10);
    plt.xlabel('K-Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    
    
    
def run2(m):   
    data = csv_to_point_label("Iris/Iris.csv", [0,1,2,3], 4)
    train_set, test_set = split_train_test(data, 0.80)
    #train_set=data[1:int(len(data)*0.8)]
    #test_set=data[int(len(data)*0.8)+1:len(data)]
    optimal_k, min_error, k_error_rates = get_optimum_k(train_set, test_set,m,
                                             k_values=list(range(2,10)))
    print("Using own KNN Algorithm and Distance Measure : "+m)
    print('For k=',optimal_k,'Validation Accurcy = ',
          round(1-min_error,3))  
    k_acc=[(i[0],1-i[1]) for i in k_error_rates]
    plt.plot(*zip(*k_acc), linestyle='--',marker='o', 
             markersize=10);
    plt.xlabel('K-Neighbors')
    plt.ylabel('Error rate')
    plt.show()