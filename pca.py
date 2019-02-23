def data_clean(df):
    import numpy as np
    #print(df)
    df=df.select_dtypes(include=np.number) 
    return df

def data_norm(df):
    import numpy as np
    #print(df)
    d_m=np.mean(df)
    d_s=np.std(df)
    d_n=(df-d_m)/d_s
    #print(d_n)
    d_n=d_n.dropna(axis=1)
    #print(d_n)
    return d_n

def data_eig(df):
    import numpy as np
    eig=np.linalg.eig
    #print(df)
    d_c=np.cov(df,rowvar=False) 
    eva,evec=eig(d_c) 
    j=eva.argsort()[::-1]
    #print(eva)
    ev=evec[:,j]
    #print(ev)
    return ev

def error_plot(df,v):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error as mse 
    err=[]
    for i in range(len(v)):   
        gt=np.asmatrix(v[:,0:i+1]).T 
        t=np.matmul(gt.T,np.matmul(gt,np.asmatrix(df).T))
        err.append(mse(np.asmatrix(df).T,t)) 
    
    n=min(df.shape)+1
    plt.title('Error Graph')
    plt.xlabel('PC')
    plt.ylabel('SSE')
    #print(err)
    return plt.plot(list(range(1,n)),err,color='blue', marker='o',markerfacecolor='red', markersize=8)

def pca(df):
    data=data_clean(df)
    norm_data=data_norm(data)
    v=data_eig(norm_data)
    error_plot(norm_data,v)