import numpy as np
import pandas as pd
from sklearn import preprocessing
class K_Means:

    def __init__(self, k=5, tol=1e-4, max_iter=1000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        randi=[]
        for i in range(self.k):
            self.centroids[i] = data[np.random.randint(0,24998)]
            #self.centroids[i] = data[i]
        for i in range(self.max_iter):
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                #print(distances)
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            #print(distances)
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False
                    break

            if optimized:
                break


    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        return distances.index(min(distances))


def run(df):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    import numpy as np
    from numpy.testing import assert_array_almost_equal
    Xtrain=df.iloc[:,0:-1].values
    Xtrain = StandardScaler().fit_transform(Xtrain)
    l=[]

    pca = PCA(n_components=14)
    principalComponents = pca.fit_transform(Xtrain)
    #print(principalComponents.shape)
    X_projected = pca.inverse_transform(principalComponents)
    loss = ((Xtrain - X_projected) ** 2).mean()
    l.append(loss)
    print("Reconstruction Error : "+str(l))
    #print(type(principalComponents))
    X=principalComponents
    y=df["xAttack"]
    clf = K_Means()
    clf.fit(X)

    correct = 0
    pred=[]
    cluster=[[],[],[],[],[]]
    for i in range(len(X)):
        toPredict = np.array(X[i].astype(float))
        toPredict = toPredict.reshape(-1, len(toPredict))
        prediction = clf.predict(toPredict)
        cluster[prediction].append(i)
        pred.append(prediction)
        #if prediction == y[i]:
        #	correct += 1

    #accuracy = max(correct/len(X), 1 - (correct/len(X)))
    #print('Accuracy', accuracy)
    #print(pred)
    y_ac=[[],[],[],[],[]]
    for i in range(5):
        print("Number of elements in Cluster "+str(i)+" : "+str(len(cluster[i])))
        for j in cluster[i]:
            y_ac[i].append(y[j])
    probe=[]
    r2l=[]
    u2r=[]
    normal=[]
    dos=[]
    clusters=["probe","r2l","u2r","normal","dos"]
    cluster_max=[]
    for i in range(5):
        print("Cluster "+str(i)+" Purity")
        score=[]
        score.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        probe.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        r2l.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        u2r.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        normal.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        dos.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        
        print(max(score))
        
        import matplotlib.pyplot as plt
        labels = clusters
        sizes = score
        colors = ['gold', 'yellowgreen', 'red', 'lightskyblue','lightcoral']
        explode = (0.1, 0, 0, 0,0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()
        cluster_max.append(clusters[score.index(max(score))])
        
def run2(df):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    import numpy as np
    from numpy.testing import assert_array_almost_equal
    Xtrain=df.iloc[:,0:-1].values
    Xtrain = StandardScaler().fit_transform(Xtrain)
    l=[]

    pca = PCA(n_components=14)
    principalComponents = pca.fit_transform(Xtrain)
    #print(principalComponents.shape)
    X_projected = pca.inverse_transform(principalComponents)
    loss = ((Xtrain - X_projected) ** 2).mean()
    l.append(loss)
    print("Reconstruction Error : "+str(l))
    #print(type(principalComponents))
    X=principalComponents
    y=df["xAttack"]
    from sklearn.cluster import KMeans,AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    cluster1 = KMeans(n_clusters=5)  
    #cluster1=GaussianMixture(n_components=5)
    cluster1.fit(X)
    correct = 0
    pred=[]
    cluster=[[],[],[],[],[]]
    for i in range(len(X)):
        toPredict = np.array(X[i].astype(float))
        toPredict = toPredict.reshape(-1, len(toPredict))
        prediction = cluster1.predict(toPredict)
        #print(prediction)
        cluster[prediction[0]].append(i)
        pred.append(prediction)
        #if prediction == y[i]:
        #	correct += 1

    #accuracy = max(correct/len(X), 1 - (correct/len(X)))
    #print('Accuracy', accuracy)
    #print(pred)
    y_ac=[[],[],[],[],[]]
    for i in range(5):
        print("Number of elements in Cluster "+str(i)+" : "+str(len(cluster[i])))
        for j in cluster[i]:
            y_ac[i].append(y[j])
    probe=[]
    r2l=[]
    u2r=[]
    normal=[]
    dos=[]
    clusters=["probe","r2l","u2r","normal","dos"]
    cluster_max=[]
    for i in range(5):
        print("Cluster "+str(i)+" Purity")
        score=[]
        score.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        probe.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        r2l.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        u2r.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        normal.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        dos.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        print(max(score))
        cluster_max.append(clusters[score.index(max(score))])
        import matplotlib.pyplot as plt
        labels = clusters
        sizes = score
        colors = ['gold', 'yellowgreen', 'red', 'lightskyblue','lightcoral']
        explode = (0.1, 0, 0, 0,0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()
        cluster_max.append(clusters[score.index(max(score))])
        
def run3(df):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    import numpy as np
    from numpy.testing import assert_array_almost_equal
    Xtrain=df.iloc[:,0:-1].values
    Xtrain = StandardScaler().fit_transform(Xtrain)
    l=[]

    pca = PCA(n_components=14)
    principalComponents = pca.fit_transform(Xtrain)
    #print(principalComponents.shape)
    X_projected = pca.inverse_transform(principalComponents)
    loss = ((Xtrain - X_projected) ** 2).mean()
    l.append(loss)
    print("Reconstruction Error : "+str(l))
    #print(type(principalComponents))
    X=principalComponents
    y=df["xAttack"]
    from sklearn.cluster import KMeans,AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    #cluster1 = KMeans(n_clusters=5)  
    cluster1=GaussianMixture(n_components=5)
    cluster1.fit(X)
    correct = 0
    pred=[]
    cluster=[[],[],[],[],[]]
    for i in range(len(X)):
        toPredict = np.array(X[i].astype(float))
        toPredict = toPredict.reshape(-1, len(toPredict))
        prediction = cluster1.predict(toPredict)
        #print(prediction)
        cluster[prediction[0]].append(i)
        pred.append(prediction)
        #if prediction == y[i]:
        #	correct += 1

    #accuracy = max(correct/len(X), 1 - (correct/len(X)))
    #print('Accuracy', accuracy)
    #print(pred)
    y_ac=[[],[],[],[],[]]
    for i in range(5):
        print("Number of elements in Cluster "+str(i)+" : "+str(len(cluster[i])))
        for j in cluster[i]:
            y_ac[i].append(y[j])
    probe=[]
    r2l=[]
    u2r=[]
    normal=[]
    dos=[]
    clusters=["probe","r2l","u2r","normal","dos"]
    cluster_max=[]
    for i in range(5):
        print("Cluster "+str(i)+" Purity")
        score=[]
        score.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        probe.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        r2l.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        u2r.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        normal.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        dos.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        print(max(score))
        cluster_max.append(clusters[score.index(max(score))])
        import matplotlib.pyplot as plt
        labels = clusters
        sizes = score
        colors = ['gold', 'yellowgreen', 'red', 'lightskyblue','lightcoral']
        explode = (0.1, 0, 0, 0,0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()
        cluster_max.append(clusters[score.index(max(score))])
        
        
def run4(df):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    import numpy as np
    from numpy.testing import assert_array_almost_equal
    Xtrain=df.iloc[:,0:-1].values
    Xtrain = StandardScaler().fit_transform(Xtrain)
    l=[]

    pca = PCA(n_components=14)
    principalComponents = pca.fit_transform(Xtrain)
    #print(principalComponents.shape)
    X_projected = pca.inverse_transform(principalComponents)
    loss = ((Xtrain - X_projected) ** 2).mean()
    l.append(loss)
    print("Reconstruction Error : "+str(l))
    #print(type(principalComponents))
    X=principalComponents
    y=df["xAttack"]
    from sklearn.cluster import KMeans,AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    cln=AgglomerativeClustering(n_clusters=5)
    cln.fit(X)
    y1=cln.fit_predict(X)
    for i in range(5):
        print(list(y1).count(i))

    correct = 0
    pred=[]
    cluster=[[],[],[],[],[]]
    for i in range(len(y1)):
        cluster[y1[i]].append(i)
    y_ac=[[],[],[],[],[]]
    for i in range(5):
        print("Number of elements in Cluster "+str(i)+" : "+str(len(cluster[i])))
        for j in cluster[i]:
            y_ac[i].append(y[j])
    probe=[]
    r2l=[]
    u2r=[]
    normal=[]
    dos=[]
    clusters=["probe","r2l","u2r","normal","dos"]
    cluster_max=[]
    for i in range(5):
        print("Cluster "+str(i)+" Purity")
        score=[]
        score.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        probe.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        r2l.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        u2r.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        normal.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        dos.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        print(max(score))
        cluster_max.append(clusters[score.index(max(score))])    
        import matplotlib.pyplot as plt
        labels = clusters
        sizes = score
        colors = ['gold', 'yellowgreen', 'red', 'lightskyblue','lightcoral']
        explode = (0.1, 0, 0, 0,0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()
        cluster_max.append(clusters[score.index(max(score))])    
