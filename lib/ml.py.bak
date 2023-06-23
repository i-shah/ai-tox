
from flask import Flask, jsonify, request, render_template, abort, make_response,url_for
from bson import json_util
from bson.objectid import ObjectId

import pymongo
from util.misc import *
import re,math
import pandas as pd
import numpy as np
import pickle
from scipy.spatial.distance import pdist,squareform,cdist

from bio.data.genra import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.model_selection import StratifiedKFold,LeaveOneOut
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import copy,warnings

warnings.simplefilter('ignore')

def getClassifiers0():
    Classifiers0 = dict(LDA = LinearDiscriminantAnalysis(),
                        #QDA=QDA(),
                        NB= GaussianNB(), 
                        KNN0=KNeighborsClassifier(3), 
                        KNN1=KNeighborsClassifier(algorithm='auto',n_neighbors=5,p=2,weights='uniform'), 
                        SVCL0=SVC(kernel='linear'), 
                        SVCR0=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                                  gamma=10.0, kernel='rbf', max_iter=100, probability=False, shrinking=True,
                                  tol=0.001, verbose=False),                    
                        CART0=DecisionTreeClassifier(max_depth=10),
                        CART1=DecisionTreeClassifier(max_features='auto'),
                        RF0=RandomForestClassifier()
                       )
    return Classifiers0

def getClassifiers2():
    Classifiers0 = dict(KNN=KNeighborsClassifier(algorithm='auto',n_neighbors=5,p=2,weights='uniform'), 
                        SVCR=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, 
                                 degree=3,gamma=10.0, kernel='rbf', max_iter=100, 
                                 probability=False, shrinking=True,
                                 tol=0.001, verbose=False),                    
                        RF=RandomForestClassifier(),
                        XGB=GradientBoostingClassifier(n_estimators=200,max_depth=3,
                                                       subsample=0.8,learning_rate=0.01,
                                                       )
                       )
    return Classifiers0

def  runKFCV(Data_cv,y,
           Data_ext=pd.DataFrame(),
           n_ds = 10,dt=None,
           cv_nfolds=10,cv_iters=5,
           CLF=None,
           DS_top=None,
           rs=202
          ):  
    """
    Data_cv: Data from cross validation
    y      : Column of Data_cv with classification attribute
    n_ds   : number of top descriptors to use
    """
    if not CLF: CLF=getClassifiers()
    # Data For CV 
    X1,Y1 = Data_cv.drop(y,axis=1),Data_cv[y]
    ID_cv = X1.index
    cls_ds=y
    Neg = Data_cv.index[Data_cv[cls_ds]==0]
    Pos = Data_cv.index[Data_cv[cls_ds]==1]
    n_n = len(Neg)
    n_p = len(Pos)
    
    if DS_top.shape[0]==0: DS_top=pd.DataFrame(np.zeros(X1.shape[1]),index=X1.columns)

    
    # Data For external validation
    ext_val = Data_ext.shape[0]>0
    if ext_val:
        X_e, Y_e = Data_ext.drop(y,axis=1),Data_ext[y] 
        ID_e     = X_e.index
        Pred_Ext = pd.DataFrame(np.zeros((len(ID_e),len(CLF))),index=ID_e,columns=CLF.keys())
    
    FS   = SelectKBest(f_classif)    # Feature selection
    Perf = []                        # Performance
    
    # Iterate through cross-validation
    for k_cv in range(cv_iters):
        SKF = StratifiedKFold(n_splits=cv_nfolds,random_state=rs)
        k_fold=0
        
        for I1,I2 in SKF.split(X1,Y1):
            k_fold+=1
            I_train,I_test=X1.index[I1],X1.index[I2]
            FS.set_params(k=n_ds)
            FS.fit_transform(X1.ix[I_train],y=Y1.ix[I_train])
            
            DS=X1.columns[FS.get_support()]
            DS_top.ix[DS]+=1
            
            X_train,Y_train = X1.ix[I_train,DS],Y1.ix[I_train]
            X_test, Y_test  = X1.ix[I_test,DS], Y1.ix[I_test]
             
            for Nm,Clf0 in CLF.iteritems():
                Clf = copy.deepcopy(Clf0)
                Clf.fit(X_train,Y_train)
                try:
                    Y_pred=Clf.predict(X_test)
                    if ext_val: Y_e_pred=Clf.predict(X_e[DS])
                except:
                    print " >> Failed",Nm
                else:
                    P=dict(pred=y,dt=dt,lr=Nm,n_ds=n_ds,
                           n_train=len(I_train),n_test=len(I_test),
                           n_obs=X1.shape[0],cvk=10,n_neg=n_n,n_pos=n_p)
                    P.update(evalPred(Y_test,Y_pred))
                    Perf.append(P)
                    if ext_val:
                        P=dict(pred=tox,dt=dt,lr=Nm,n_ds=n_ds,n_train=len(I_train),n_test=X_e.shape[0],
                               n_obs=len(I_train)+X_e.shape[0],
                                pt='cvt_ext',
                               n_neg=n_n,n_pos=n_p)
                        Y_e_pred=Clf.predict(X_e)
                        P.update(evalPred(Y_e,Y_e_pred))
                        Perf.append(P)
    return Perf


def  runLooCV(Data_cv,y,
              n_ds = 10,dt=None,
              cv_nfolds=10,cv_iters=5,
              CLF=None,
              DS_top=None,
              dbg=False
              ):  
    """
    Data_cv: Data from cross validation
    y      : Column of Data_cv with classification attribute
    n_ds   : number of top descriptors to use
    """
    if not CLF: CLF=getClassifiers()
    # Data For CV 
    X1,Y1 = Data_cv.drop(y,axis=1),Data_cv[y]
    ID_cv = X1.index
    cls_ds=y
    Neg = Data_cv.index[Data_cv[cls_ds]==0]
    Pos = Data_cv.index[Data_cv[cls_ds]==1]
    n_n = len(Neg)
    n_p = len(Pos)
    
    if DS_top.shape[0]==0: DS_top=pd.DataFrame(np.zeros(X1.shape[1]),index=X1.columns)

    
    # Data For external validation
    
    
    FS   = SelectKBest(f_classif)    # Feature selection
    Perf = []                        # Performance
    
    # Iterate through cross-validation
    for k_cv in range(cv_iters):
        LOO = LeaveOneOut()
        k_fold=0
        for I_tr,I_te in LOO.split(X1):
            k_fold+=1
            I_train,I_test=X1.index[I_tr],X1.index[I_te]
            FS.set_params(k=n_ds)
            FS.fit_transform(X1.ix[I_train],y=Y1.ix[I_train])
            
            DS=X1.columns[FS.get_support()]
            DS_top.ix[DS]+=1
            
            X_train,Y_train = X1.ix[I_train,DS],Y1.ix[I_train]
            X_test, Y_test  = X1.ix[I_test,DS], Y1.ix[I_test]
            
            for Nm,Clf0 in CLF.iteritems():
                Clf = copy.deepcopy(Clf0)
                Clf.fit(X_train,Y_train)
                try:
                    Y_pred=Clf.predict(X_test)
                except:
                    print " >> Failed",Nm
                else:
                    P=dict(pred=y,dt=dt,lr=Nm,n_ds=n_ds,
                           n_train=len(I_train),n_test=len(I_test),
                           n_obs=X1.shape[0],n_neg=n_n,n_pos=n_p)
                    P.update(evalPred(Y_test,Y_pred))
                    Perf.append(P)

    return Perf

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer,f1_score,accuracy_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold

def  runXGB(Data_cv,y,
            Data_ext=pd.DataFrame(),
            dt=None,
            cv_nfolds=10,cv_iters=5,
            DS_top=None,
            rs=202,
            gs_scorer='f1',
            gs_nj=30,
            xgb_params=dict(loss='deviance',
                            learning_rate=0.01,
                            max_depth=15,
                            subsample=0.8,
                            n_estimators=400)
          ):  
    """
    Data_cv: Data from cross validation
    y      : Column of Data_cv with classification attribute
    n_ds   : number of top descriptors to use
    """
    # Data For CV 
    X1,Y1 = Data_cv.drop(y,axis=1),Data_cv[y]
    ID_cv = X1.index
    cls_ds=y
    Neg = Data_cv.index[Data_cv[cls_ds]==0]
    Pos = Data_cv.index[Data_cv[cls_ds]==1]
    n_n = len(Neg)
    n_p = len(Pos)
    
    Perf = []                        # Performance
    # Classifier
    Clf = GradientBoostingClassifier(**xgb_params)
    Nm  = 'XGB'
    # Iterate through cross-validation
    for k_cv in range(cv_iters):
        SKF = StratifiedKFold(n_splits=cv_nfolds,random_state=rs)
        k_fold=0
        
        for I1,I2 in SKF.split(X1,Y1):
            k_fold+=1
            I_train,I_test=X1.index[I1],X1.index[I2]
            
            X_train,Y_train = X1.ix[I_train],Y1.ix[I_train]
            X_test, Y_test  = X1.ix[I_test], Y1.ix[I_test]
             
            
            Clf.fit(X_train,Y_train)
            try:
                Y_pred=Clf.predict(X_test)
            except:
                print " >> Failed",Nm
            else:
                P=dict(pred=y,dt=dt,lr=Nm,n_ds=X1.shape[1],
                       n_train=len(I_train),n_test=len(I_test),
                       n_obs=X1.shape[0],cvk=cv_nfolds,n_neg=n_n,n_pos=n_p)
                P.update(evalPred(Y_test,Y_pred))
                Perf.append(P)

    return Perf

def  runXGBGS(Data_cv,y,
            Data_ext=pd.DataFrame(),
            dt=None,
            cv_nfolds=10,cv_iters=5,
            DS_top=None,
            rs=202,
            gs_scorer='f1',
            gs_nj=30
          ):  
    """
    Data_cv: Data from cross validation
    y      : Column of Data_cv with classification attribute
    n_ds   : number of top descriptors to use
    """
    # Data For CV 
    X1,Y1 = Data_cv.drop(y,axis=1),Data_cv[y]
    ID_cv = X1.index
    cls_ds=y
    Neg = Data_cv.index[Data_cv[cls_ds]==0]
    Pos = Data_cv.index[Data_cv[cls_ds]==1]
    n_n = len(Neg)
    n_p = len(Pos)
    
    if DS_top.shape[0]==0: DS_top=pd.DataFrame(np.zeros(X1.shape[1]),index=X1.columns)

    Perf = []                        # Performance
    # Classifier
    GBC = GradientBoostingClassifier(loss='deviance')
    
    params={'learning_rate': 10**np.linspace(-3,-1,num=5),
            'max_depth': range(4,16,2),
            'n_estimators': range(100,500,50)
            }
    
    scorer = None
    if gs_scorer=='f1':
        scorer = make_scorer(f1_score,greater_is_better=True)
    else:
        scorer = make_scorer(accuracy_score,greater_is_better=True)
        
    Grid1= GridSearchCV(estimator=GBC,param_grid=params,
                        n_jobs=gs_nj,cv=StratifiedKFold(n_splits=3),
                        verbose=0,
                        scoring=scorer)

    # Iterate through cross-validation
    for k_cv in range(cv_iters):
        SKF = StratifiedKFold(n_splits=cv_nfolds,random_state=rs)
        k_fold=0
        
        for I1,I2 in SKF.split(X1,Y1):
            k_fold+=1
            I_train,I_test=X1.index[I1],X1.index[I2]
            
            #DS=X1.columns[FS.get_support()]
            #DS_top.ix[DS]+=1
            
            X_train,Y_train = X1.ix[I_train,DS],Y1.ix[I_train]
            X_test, Y_test  = X1.ix[I_test,DS], Y1.ix[I_test]
             
            
            Clf = copy.deepcopy(Clf0)
            Clf.fit(X_train,Y_train)
            try:
                Y_pred=Clf.predict(X_test)
            except:
                print " >> Failed",Nm
            else:
                P=dict(pred=y,dt=dt,lr=Nm,n_ds=n_ds,
                       n_train=len(I_train),n_test=len(I_test),
                       n_obs=X1.shape[0],cvk=10,n_neg=n_n,n_pos=n_p)
                P.update(evalPred(Y_test,Y_pred))
                Perf.append(P)

    return Perf


def evalPred(Y_truth,Y_inferred,post=None):
    M = dict(sens=recall_score(Y_truth,Y_inferred),
             spec=precision_score(Y_truth,Y_inferred),
             acc=accuracy_score(Y_truth,Y_inferred),
             f1=f1_score(Y_truth,Y_inferred))
       
    M['bacc']=0.5*(M['sens']+M['spec'])
   
    if post: M={k+post:v for k,v in M.iteritems()}
    
    return M

def summarizePerf(P):
    P0=pd.DataFrame(P)
    Agg = dict(bacc=dict(mn=np.mean,sd=np.std),
               f1=dict(mn=np.mean,sd=np.std),
               acc=dict(mn=np.mean,sd=np.std),
               sens=dict(mn=np.mean,sd=np.std),
               spec=dict(mn=np.mean,sd=np.std),
               n_train=dict(mn=np.median),
               n_test =dict(mn=np.median)
              )
    P1=P0.groupby(by=['lr','pt','cv_kfold']).aggregate(Agg)
    P2=P1.reset_index()
    C=['_'.join([k for k in i if k]) for i in P2.columns]
    P2.columns=C
    
    return P2

