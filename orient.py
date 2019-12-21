# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:42:44 2019

@author: jashj
"""

from NeuralNetwork import NeuralNetwork as nn
from KNN import KNN as knn
from DTree import dtreemain
import pandas as pd
import numpy as np
import pickle
import sys

def orient(name, filename, model_file, model):
    
    if name=='train':
        
        if model=='nearest' or model=='best':
            train=pd.read_csv(filename,sep=' ',header=None)
            filename_knn=model_file
            file=open(filename_knn,'wb')
            pickle.dump(train,file)
        
        if model=='nnet':
            train=pd.read_csv(filename,sep=' ',header=None) 
            x_train=train.drop(columns=[0,1],axis=1)
            y_train=train[1]
            y_train=pd.get_dummies(y_train)
            y_columns=y_train.columns
            print(x_train.shape[0], 'train samples')
            a=nn(25,0.001,0.9)
            (w1,w2,w3,b1,b2,b3)=a.fit(x_train,y_train)
            
            weights={'w1':w1,
                     'w2':w2,
                     'w3':w3,
                     'b1':b1,
                     'b2':b2,
                     'b3':b3}
            filename_nn=model_file
            file=open(filename_nn,'wb')
            pickle.dump(weights,file)
    
        if model =='tree':
            dtreemain(name, filename, model_file)
            
    if name=='test':
        
        if model=='nearest' or model == 'best':
            file=open(model_file,'rb')
            train=pickle.load(file)
            test=pd.read_csv(filename, sep=' ',header=None)
            X_test=test.drop(columns=[0,1],axis=1)
            y_filenames=test[0]
            y_test=test[1]
            X_test=X_test.to_numpy()
            y_test=y_test.to_numpy()
            obj=knn(10)
            ypred=obj.predict(train,X_test)
            f= open("output K Nearest Neighbors.txt","w")
            for i in range(len(X_test)):
                with open('output K Nearest Neighbors.txt','a') as f:
                        f.write(str(y_filenames[i])+ ' '+str(ypred[i])+'\n')
            pass
        if model=='tree' :
            dtreemain(name, filename, model_file)
        
        if model=='nnet':        
    
            test=pd.read_csv(filename,sep=' ',header=None)    
            x_test=test.drop(columns=[0,1],axis=1)
            y_test=test[1]
            y_filenames=test[0]
            y_test=pd.get_dummies(y_test)
            print(x_test.shape[0], 'test samples')
            file=open(model_file,'rb')
            new_weights=pickle.load(file)
            w1f=new_weights['w1']
            w2f=new_weights['w2']
            w3f=new_weights['w3']
            b1f=new_weights['b1']
            b2f=new_weights['b2']
            b3f=new_weights['b3']
            
            y_test_predicted=a.predict(x_test,w1f,w2f,w3f,b1f,b2f,b3f)   
            zero_one=(y_test_predicted == y_test_predicted.max(axis=1)[:,None]).astype(int)        
            
            diff=(y_test == zero_one).sum(axis=1)
            accuracy=np.count_nonzero(diff == y_test.shape[1])
            print('accuracy ',accuracy/diff.shape[0]*100) 
            f= open("Output Neural Network.txt","w")
            for i in range(len(x_test)):
                with open('Output Neural Network.txt','a') as f:
                    f.write(str(y_filenames[i])+ ' '+str(y_columns[np.argmax(y_test_predicted[i])])+'\n')
        
        
if __name__== "__main__":
    
    if(len(sys.argv) != 5):
        raise Exception('Error: expected 4 command line arguments') 
    orient(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        
        
        
        
        
        
        
        
        
        