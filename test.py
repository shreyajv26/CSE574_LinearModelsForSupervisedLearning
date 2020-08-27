import pickle
import numpy as np

def logisticObjVal(w, X, y):
    print ("In function call")
    # compute log-loss error (scalar) with respect
    # to w (vector) for the given data X and y                               
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # error = scalar
    sum=0.0
    for i in range(np.size(Xtrain_i,1)):
        data_point=Xtrain_i[i]
        temp=0
        for j in range(len(data_point)):
            
            print ("i")
            print (i)
            print ("j")
            print (j)
            temp=temp+(Xtrain_i[i][j]*w[i][0])
        sum=sum+(np.log(1+np.exp(y[i][0]*temp*-1)))
    
    error=sum/(np.size(Xtrain_i,0))
        
            
        #get column for that data point
        
        #multiply elementwise to weight matrix
        
    
    if len(w.shape) == 1:
        w = w[:,np.newaxis]
    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    print ("Error"+str(error))
    return error



Xtrain,ytrain, Xtest, ytest = pickle.load(open('sample.pickle','rb')) 
print (type(Xtrain))
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
print ("Size of rows "+str(np.size(Xtrain_i,0)))

args = (Xtrain_i,ytrain)
opts = {'maxiter' : 50}    # Preferred value.    
w_init = np.zeros((Xtrain_i.shape[1],1))
logisticObjVal(w_init,Xtrain_i,ytrain)



