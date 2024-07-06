import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import model.model as mm
import torch.utils.data as Data
import torch
import torch.nn as nn
import os
from com.utils import plot_ROC,plot_Multi_ROC,plot_five_ROC
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import ElasticNetCV,ElasticNet
from sklearn.linear_model import LogisticRegression
from scipy import stats
from model.multi_scale_model import *
import model.model as mm
from model.multi_scale_CGM import *
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import norm
from scipy.interpolate import lagrange
from sklearn.model_selection import train_test_split,KFold,RepeatedKFold
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
def Lagrange_interp(xdata,ydata,x):

    v = np.zeros(x.size) #插值结果数值
    n = xdata.size
    for k in range(n): #n个Lagrange多项式
        w = np.ones(x.size) #记录连乘
        for j in range(n):
            if j != k:
                w *= (x-xdata[j])/(xdata[k]-xdata[j])
        v += w*ydata[k] # L_k(x)乘以对应函数值f(x_k)
    return v

def AUC_CI(auc, label, alpha = 0.05):
    label = np.array(label)#防止label不是array类型
    n1, n2 = np.sum(label == 1), np.sum(label == 0)
    q1 = auc / (2-auc)
    q2 = (2 * auc ** 2) / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 -1) * (q2 - auc ** 2)) / (n1 * n2))
    confidence_level = 1 - alpha
    z_lower, z_upper = norm.interval(confidence_level)
    lowerb, upperb = auc + z_lower * se, auc + z_upper * se
    return (lowerb, upperb)

def bootstrap_auc(data_df,data_cgm_df):
    algorithm_classes = 6
    bootstraps = 100
    statistics_auc = np.zeros((algorithm_classes, bootstraps))
    statistics_accuracy = np.zeros((algorithm_classes, bootstraps))
    statistics_precision = np.zeros((algorithm_classes, bootstraps))
    statistics_recall = np.zeros((algorithm_classes, bootstraps))

    statistics_auc_percen = np.zeros((algorithm_classes, 2))
    statistics_acc_percen = np.zeros((algorithm_classes, 2))
    statistics_pre_percen = np.zeros((algorithm_classes, 2))
    statistics_recall_percen = np.zeros((algorithm_classes, 2))
    thre = 0.5
    for index in range(algorithm_classes):
        if index <5:
            fold_size = data_df.shape[0]
            df = pd.DataFrame(columns=['y', 'pred'])
            df.loc[:, 'y'] =data_df[data_df.columns[index*2]].values
            df.loc[:, 'pred'] = data_df[data_df.columns[index*2+1]].values
            #print(data_df.columns[index*2],data_df.columns[index*2+1])
        else:
            fold_size = data_cgm_df.shape[0]
            df = pd.DataFrame(columns=['y', 'pred'])
            df.loc[:, 'y'] =data_cgm_df[data_cgm_df.columns[0]].values
            df.loc[:, 'pred'] = data_cgm_df[data_cgm_df.columns[1]].values
            #print(data_cgm_df.columns[0],data_cgm_df.columns[1])
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            fpr,tpr,threshold= metrics.roc_curve(y_sample, pred_sample)
            roc_auc=metrics.auc(fpr,tpr)
            pred_sample[pred_sample>=thre]=1
            pred_sample[pred_sample<thre]=0
            accuracy = metrics.accuracy_score(y_sample,pred_sample)
            precision = metrics.precision_score(y_sample,pred_sample)
            recall = metrics.recall_score(y_sample,pred_sample)
            statistics_auc[index][i] = roc_auc
            statistics_accuracy[index][i] = accuracy
            statistics_precision[index][i] = precision
            statistics_recall[index][i] = recall
        statistics_auc_percen[index,0] =  np.percentile(statistics_auc[index,:],2.5)
        statistics_auc_percen[index,1] =  np.percentile(statistics_auc[index,:],97.5) 

        statistics_acc_percen[index,0] =  np.percentile(statistics_accuracy[index,:],2.5)
        statistics_acc_percen[index,1] =  np.percentile(statistics_accuracy[index,:],97.5)  

        statistics_pre_percen[index,0] =  np.percentile(statistics_precision[index,:],2.5)
        statistics_pre_percen[index,1] =  np.percentile(statistics_precision[index,:],97.5)  

        statistics_recall_percen[index,0] =  np.percentile(statistics_recall[index,:],2.5)
        statistics_recall_percen[index,1] =  np.percentile(statistics_recall[index,:],97.5)     

    return statistics_auc_percen,statistics_acc_percen,statistics_pre_percen,statistics_recall_percen
if __name__ == '__main__':
    data = pd.read_csv('./dataset.csv',sep=',', header= None)
    data = np.array(data)
    random_num=200
    index_fusion = 4
    index_CGM = 7
    index_clinical = 4
    # ------------------fusion model----------------
    model_path ='./results/fusion/best_model.pt'
    mlp_fusion = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=1)
    mlp_fusion.load_state_dict(torch.load(model_path))

    # ------------------CGM model--------------------
    model_path ='./results/CGM_total/best_model.pt'
    mlp_CGM = MSResNet_CGM(input_channel=1, layers=[1, 1, 1, 1], num_classes=1)
    mlp_CGM.load_state_dict(torch.load(model_path))
    #-------------------Clinical model---------------5-fold/
    model_path ='./results/clinical/best_model.pt'
    mlp_clinical = mm.MLP()
    mlp_clinical.load_state_dict(torch.load(model_path))
    #=============================preprocessing===========================
    dim=290
    for ind in range(data.shape[0]):
        data[ind,2:dim]=(data[ind,2:dim]-np.min(data[ind,2:dim]))/(np.max(data[ind,2:dim])-np.min(data[ind,2:dim]))
    for ind in range(15):
        data[:,290+ind]=(data[:,290+ind]-np.min(data[:,290+ind]))/(np.max(data[:,290+ind])-np.min(data[:,290+ind]))
    
    # for ind in range(data.shape[0]):
    #     data[ind,2:dim]=preprocessing.MinMaxScaler().fit_transform(data[ind,2:dim].reshape(-1,1)).reshape(-1)
    # for ind in range(16):
    #     data[:,290+ind]=preprocessing.MinMaxScaler().fit_transform(data[:,290+ind].reshape(-1,1)).reshape(-1)
    #==========================================fusion network================
    data = np.concatenate((data[:,1].reshape(-1,1),data[:,0].reshape(-1,1),data[:,2:]),axis=1)
    x_train,x_test,y_train,y_test=train_test_split(data[:,1:],data[:,0],test_size=0.12,random_state=random_num)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    train_dataset = np.concatenate((y_train,x_train),axis=1)
    test_dataset = np.concatenate((y_test,x_test),axis=1) 

    p=1
    kfold=10
    
    kf = RepeatedKFold(n_splits=kfold,n_repeats=p,random_state=random_num)
    kf_list = list(kf.split(train_dataset))

    kf_train,kf_val = kf_list[index_fusion]
    x_train_fusion = train_dataset[kf_train,:].astype(None)
    y_train_fusion = train_dataset[kf_train,0].astype(None)
    x_val_fusion = train_dataset[kf_val,:].astype(None)
    y_val_fusion = train_dataset[kf_val,0].astype(None)
    df_train = pd.DataFrame(x_train_fusion)
    df_train.to_excel('train_data.xlsx',index=False)
    df_val = pd.DataFrame(x_val_fusion)
    df_val.to_excel('val_data.xlsx',index=False)

    x_train_ml = train_dataset[kf_train,290:306].astype(None)
    y_train_ml = train_dataset[kf_train,0].astype(None)
    x_val_ml = train_dataset[kf_val,290:306].astype(None)
    y_val_ml = train_dataset[kf_val,0].astype(None)

    x_test_ml = test_dataset[:,290:306]
    y_test_ml = test_dataset[:,0]

    kf_train,kf_val = kf_list[index_CGM]
    x_train_CGM = train_dataset[kf_train,:].astype(None)
    y_train_CGM = train_dataset[kf_train,0].astype(None)
    x_val_CGM = train_dataset[kf_val,:].astype(None)
    y_val_CGM = train_dataset[kf_val,0].astype(None)

    kf_train,kf_val = kf_list[index_clinical]
    x_train_clinical = train_dataset[kf_train,:].astype(None)
    y_train_clinical = train_dataset[kf_train,0].astype(None)

    x_val_clinical = train_dataset[kf_val,:].astype(None)
    y_val_clinical = train_dataset[kf_val,0].astype(None)

    x_test = test_dataset[:,:]
    y_test = test_dataset[:,0]

    df_test = pd.DataFrame(x_test)
    df_test.to_excel('test_data.xlsx',index=False)
    #============================ train dataset ===================================
    train_data_fusion = Data.TensorDataset(torch.FloatTensor(x_train_fusion),torch.Tensor(y_train_fusion))
    train_loader_fusion = Data.DataLoader(dataset=train_data_fusion,batch_size= x_train_fusion.shape[0], shuffle = False)

    train_data_CGM = Data.TensorDataset(torch.FloatTensor(x_train_CGM),torch.Tensor(y_train_CGM))
    train_loader_CGM = Data.DataLoader(dataset=train_data_CGM,batch_size= x_train_CGM.shape[0], shuffle = False)

    train_data_clinical = Data.TensorDataset(torch.FloatTensor(x_train_clinical),torch.Tensor(y_train_clinical))
    train_loader_clinical = Data.DataLoader(dataset=train_data_clinical,batch_size= x_train_clinical.shape[0], shuffle = False)

    #============================ validation dataset ===================================
    val_data_fusion = Data.TensorDataset(torch.FloatTensor(x_val_fusion),torch.Tensor(y_val_fusion))
    val_loader_fusion = Data.DataLoader(dataset=val_data_fusion,batch_size= x_val_fusion.shape[0], shuffle = False)

    val_data_CGM = Data.TensorDataset(torch.FloatTensor(x_val_CGM),torch.Tensor(y_val_CGM))
    val_loader_CGM = Data.DataLoader(dataset=val_data_CGM,batch_size= x_val_CGM.shape[0], shuffle = False)

    val_data_clinical = Data.TensorDataset(torch.FloatTensor(x_val_clinical),torch.Tensor(y_val_clinical))
    val_loader_clinical = Data.DataLoader(dataset=val_data_clinical,batch_size= x_val_clinical.shape[0], shuffle = False)
    #============================ test dataset ===================================
    test_data = Data.TensorDataset(torch.FloatTensor(x_test),torch.Tensor(y_test))
    test_loader = Data.DataLoader(dataset=test_data,batch_size= x_test.shape[0], shuffle = False)
    mlp_fusion.eval()
    mlp_CGM.eval()
    mlp_clinical.eval()
    result_path = "./results"
    feature = np.zeros([x_train_CGM.shape[0],14])
    N=x_train_CGM.shape[0]
    #=============================FUSION MODEL===============================
    pred_fusion_train = []
    gt_fusion_train = []
    with torch.no_grad():
        for step,(data_batch,label) in enumerate(train_loader_fusion):
            output,featuremap=mlp_fusion(data_batch)
            output = torch.sigmoid(output) 
            pred_fusion_train.append(output.view(-1).tolist())
            gt_fusion_train.append(label.tolist())
        pred_fusion_train = torch.tensor(pred_fusion_train).view(-1)
        gt_fusion_train = torch.tensor(gt_fusion_train).view(-1)

    pred_fusion_test=[]
    gt_fusion_test =[]
    pred_fusion_val =[]
    gt_fusion_val =[]
    with torch.no_grad():
        for step, (data_batch,label) in enumerate(test_loader): 
            output,featuremap = mlp_fusion(data_batch) 
            output = torch.sigmoid(output)     
            pred_fusion_test.append(output.view(-1).tolist())
            gt_fusion_test.append(label.tolist()) 
        pred_fusion_test= torch.tensor(pred_fusion_test).view(-1)
        gt_fusion_test= torch.tensor(gt_fusion_test).view(-1)
    
    with torch.no_grad():
        for step, (data_batch,label) in enumerate(val_loader_fusion): 
            output,featuremap = mlp_fusion(data_batch) 
            output = torch.sigmoid(output)     
            pred_fusion_val.append(output.view(-1).tolist())
            gt_fusion_val.append(label.tolist()) 
        pred_fusion_val= torch.tensor(pred_fusion_val).view(-1)
        gt_fusion_val= torch.tensor(gt_fusion_val).view(-1)
    
    #=============================CGM MODEL===============================
    pred_cgm_train = []
    gt_cgm_train = []
    pred_cgm_test =[]
    gt_cgm_test =[]
    pred_cgm_val =[]
    gt_cgm_val =[]
    with torch.no_grad():
        for step,(data_batch,label) in enumerate(train_loader_CGM):
            output,featuremap=mlp_CGM(data_batch)
            output = torch.sigmoid(output) 
            pred_cgm_train.append(output.view(-1).tolist())
            gt_cgm_train.append(label.tolist())
            
            features=featuremap
            norm_ = featuremap.cpu().numpy()
            norm_=norm_.reshape(N,-1)
            featureNorm=np.linalg.norm(norm_,ord=2,axis=1)
            label=label.cpu().numpy()
            norm_0 = featureNorm[label==0]
            norm_1 = featureNorm[label==1]
        pred_cgm_train = torch.tensor(pred_cgm_train).view(-1)
        gt_cgm_train = torch.tensor(gt_cgm_train).view(-1)

    with torch.no_grad():
        for step, (data_batch,label) in enumerate(test_loader): 
            output,featuremap = mlp_CGM(data_batch) 
            output = torch.sigmoid(output)     
            pred_cgm_test.append(output.view(-1).tolist())
            gt_cgm_test.append(label.tolist()) 
        norm_ = featuremap.cpu().numpy()
        norm_=norm_.reshape(x_test.shape[0],-1)
        featureNorm=np.linalg.norm(norm_,ord=2,axis=1)
        label=label.cpu().numpy()
        norm_0 = featureNorm[label==0]
        norm_1 = featureNorm[label==1]
        t,p_test_CGM=stats.ttest_ind(norm_0,norm_1,equal_var=False,alternative="two-sided")

        pred_cgm_test= torch.tensor(pred_cgm_test).view(-1)
        gt_cgm_test= torch.tensor(gt_cgm_test).view(-1)
    with torch.no_grad():
        for step, (data_batch,label) in enumerate(val_loader_CGM): 
            output,featuremap = mlp_CGM(data_batch) 
            output = torch.sigmoid(output)     
            pred_cgm_val.append(output.view(-1).tolist())
            gt_cgm_val.append(label.tolist()) 
        
        pred_cgm_val= torch.tensor(pred_cgm_val).view(-1)
        gt_cgm_val= torch.tensor(gt_cgm_val).view(-1)
    #=============================Clinical MODEL===============================
    pred_clinical_train = []
    gt_clinical_train = []
    pred_clinical_test =[]
    gt_clinical_test=[]
    pred_clinical_val =[]
    gt_clinical_val=[]
    with torch.no_grad():
        for step,(data_batch,label) in enumerate(train_loader_clinical):
            output,featuremap=mlp_clinical(data_batch)
            output = torch.sigmoid(output) 
            pred_clinical_train.append(output.view(-1).tolist())
            gt_clinical_train.append(label.tolist())
        pred_clinical_train = torch.tensor(pred_clinical_train).view(-1)
        gt_clinical_train = torch.tensor(gt_clinical_train).view(-1)

    with torch.no_grad():
        for step, (data_batch,label) in enumerate(test_loader): 
            output,featuremap = mlp_clinical(data_batch) 
            output = torch.sigmoid(output)     
            pred_clinical_test.append(output.view(-1).tolist())
            gt_clinical_test.append(label.tolist()) 
        norm_ = featuremap.cpu().numpy()
        norm_=norm_.reshape(x_test.shape[0],-1)
        featureNorm=np.linalg.norm(norm_,ord=2,axis=1)
        label=label.cpu().numpy()
        norm_0 = featureNorm[label==0]
        norm_1 = featureNorm[label==1]
        t,p_test_CGM=stats.ttest_ind(norm_0,norm_1,equal_var=False,alternative="two-sided")

        pred_clinical_test= torch.tensor(pred_clinical_test).view(-1)
        gt_clinical_test= torch.tensor(gt_clinical_test).view(-1)
    with torch.no_grad():
        for step, (data_batch,label) in enumerate(val_loader_clinical): 
            output,featuremap = mlp_clinical(data_batch) 
            output = torch.sigmoid(output)     
            pred_clinical_val.append(output.view(-1).tolist())
            gt_clinical_val.append(label.tolist()) 
        pred_clinical_val= torch.tensor(pred_clinical_val).view(-1)
        gt_clinical_val= torch.tensor(gt_clinical_val).view(-1)
    #--------------------------------Random Forest--------------------------------
        #data=data_raw
        rf = RandomForestClassifier(n_estimators=10,max_depth=8,min_samples_leaf=5,min_samples_split=2,max_features=10,criterion="gini",random_state=10).fit(x_train_ml,y_train_ml)
        train_pred_rf = rf.predict_proba(x_train_ml)
        pred_rf_test = rf.predict_proba(x_test_ml)
        pred_rf_val = rf.predict_proba(x_val_ml)
        #-------------------------Support Vector Machine------------------------------
        svm_model = SVC(kernel="rbf",C=200,gamma="auto",probability=True,random_state=30).fit(x_train_ml, y_train_ml)
        train_pred_svm = svm_model.predict_proba(x_train_ml)
        pred_svm_test = svm_model.predict_proba(x_test_ml)
        pred_svm_val = svm_model.predict_proba(x_val_ml)
        #-------------------------------ElasticNetCV----------------------------------
        #encv_model = ElasticNet(max_iter=200,cv=10,l1_ratio=0.1).fit(x_train,y_train)
        encv_model = ElasticNetCV(l1_ratio=0.1).fit(x_train_ml,y_train_ml)
        train_pred_encv = encv_model.predict(x_train_ml)
        pred_encv_test = encv_model.predict(x_test_ml)
        pred_encv_val = encv_model.predict(x_val_ml)
        #-------------------------------lr----------------------------------
        lr_model = LogisticRegression(multi_class="multinomial", solver="newton-cg").fit(x_train_ml,y_train_ml)
        train_pred_lr = lr_model.predict_proba(x_train_ml)
        pred_lr_test = lr_model.predict_proba(x_test_ml)
        pred_lr_val = lr_model.predict_proba(x_val_ml)
        #-------------------------------decision tree----------------------------------
        dt_model=DecisionTreeClassifier(criterion='gini',random_state=30,splitter="random",min_samples_leaf=3,min_samples_split=3).fit(x_train_ml, y_train_ml)
        train_pred_dt = dt_model.predict_proba(x_train_ml)
        pred_dt_test = dt_model.predict_proba(x_test_ml)
        pred_dt_val = dt_model.predict_proba(x_val_ml)
        #------------------------------------Results----------------------------------
        save_path = os.path.join(result_path,"train_roc.png")
        auc1,auc2,auc3,auc4,auc5,auc6,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,precision1,precision2,precision3,precision4,precision5,precision6,recall1,recall2,recall3,recall4,recall5,recall6=plot_five_ROC(y_train_fusion,pred_fusion_train,pred_cgm_train,pred_clinical_train,train_pred_rf[:,1],train_pred_lr[:,1],train_pred_dt[:,1],y_train_CGM,y_train_clinical,save_path)
       
        train_data = pd.DataFrame({"y_fusion":y_train_fusion,"y_pred":pred_fusion_train,"y_clinical":y_train_clinical,"pred_clinical":pred_clinical_train,"y_rf":y_train_clinical,"pred_rf":train_pred_rf[:,1],"y_lr":y_train_clinical,"pred_lr":train_pred_lr[:,1],"y_dt":y_train_clinical,"pred_dt":train_pred_dt[:,1]})
        train_cgm = pd.DataFrame({"y_cgm":y_train_CGM,"pred_cgm":pred_cgm_train})
        auc_train,acc_train,prec_train,recall_train = bootstrap_auc(train_data,train_cgm)

        print("-----------------------------------------------------------------------Train--------------------------------------------------------------------------------")
        print("----Metric--|----------FM-----------|-----------CGM---------|------------CL---------|------------RF---------|------------LR---------|------------DT---------")
        print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("AUC",auc1,auc_train[0,0],auc_train[0,1],auc2,auc_train[5,0],auc_train[5,1],auc3,auc_train[1,0],auc_train[1,1],auc4,auc_train[2,0],auc_train[2,1],auc5,auc_train[3,0],auc_train[3,1],auc6,auc_train[4,0],auc_train[4,1]))
        print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("Accuracy",accuracy1,acc_train[0,0],acc_train[0,1],accuracy2,acc_train[5,0],acc_train[5,1],accuracy3,acc_train[1,0],acc_train[1,1],accuracy4,acc_train[2,0],acc_train[2,1],accuracy5,acc_train[3,0],acc_train[3,1],accuracy6,acc_train[4,0],acc_train[4,1]))
        #print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("Precision",precision1,prec_train[0,0],prec_train[0,1],precision2,prec_train[5,0],prec_train[5,1],precision3,prec_train[1,0],prec_train[1,1],precision4,prec_train[2,0],prec_train[2,1],precision5,prec_train[3,0],prec_train[3,1],precision6,prec_train[4,0],prec_train[4,1]))
        #print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("Recall",recall1,recall_train[0,0],recall_train[0,1],recall2,recall_train[5,0],recall_train[5,1],recall3,recall_train[1,0],recall_train[1,1],recall4,recall_train[2,0],recall_train[2,1],recall5,recall_train[3,0],recall_train[3,1],recall6,recall_train[4,0],recall_train[4,1],))
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------")
        

        save_path = os.path.join(result_path,"test_roc.png")
        auc1,auc2,auc3,auc4,auc5,auc6,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,precision1,precision2,precision3,precision4,precision5,precision6,recall1,recall2,recall3,recall4,recall5,recall6=plot_five_ROC(y_test,pred_fusion_test,pred_cgm_test,pred_clinical_test,pred_rf_test[:,1],pred_lr_test[:,1],pred_dt_test[:,1],y_test,y_test,save_path)
        test_data = pd.DataFrame({"y_fusion":y_test,"y_pred":pred_fusion_test,"y_clinical":y_test,"pred_clinical":pred_clinical_test,"y_rf":y_test,"pred_rf":pred_rf_test[:,1],"y_lr":y_test,"pred_lr":pred_lr_test[:,1],"y_dt":y_test,"pred_dt":pred_dt_test[:,1]})
        test_cgm = pd.DataFrame({"y_cgm":y_test,"pred_cgm":pred_cgm_test})
        auc_train,acc_train,prec_train,recall_train = bootstrap_auc(test_data,test_cgm)

        print("-----------------------------------------------------------------------Test---------------------------------------------------------------------------------")
        print("----Metric--|----------FM-----------|-----------CGM---------|------------CL---------|------------RF---------|------------LR---------|------------DT---------")
        print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("AUC",auc1,auc_train[0,0],auc_train[0,1],auc2,auc_train[5,0],auc_train[5,1],auc3,auc_train[1,0],auc_train[1,1],auc4,auc_train[2,0],auc_train[2,1],auc5,auc_train[3,0],auc_train[3,1],auc6,auc_train[4,0],auc_train[4,1]))
        print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("Accuracy",accuracy1,acc_train[0,0],acc_train[0,1],accuracy2,acc_train[5,0],acc_train[5,1],accuracy3,acc_train[1,0],acc_train[1,1],accuracy4,acc_train[2,0],acc_train[2,1],accuracy5,acc_train[3,0],acc_train[3,1],accuracy6,acc_train[4,0],acc_train[4,1]))
        #print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("Precision",precision1,prec_train[0,0],prec_train[0,1],precision2,prec_train[5,0],prec_train[5,1],precision3,prec_train[1,0],prec_train[1,1],precision4,prec_train[2,0],prec_train[2,1],precision5,prec_train[3,0],prec_train[3,1],precision6,prec_train[4,0],prec_train[4,1]))
        #print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("Recall",recall1,recall_train[0,0],recall_train[0,1],recall2,recall_train[5,0],recall_train[5,1],recall3,recall_train[1,0],recall_train[1,1],recall4,recall_train[2,0],recall_train[2,1],recall5,recall_train[3,0],recall_train[3,1],recall6,recall_train[4,0],recall_train[4,1],))
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------")
       
    
        save_path = os.path.join(result_path,"val_roc.png")
        auc1,auc2,auc3,auc4,auc5,auc6,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,precision1,precision2,precision3,precision4,precision5,precision6,recall1,recall2,recall3,recall4,recall5,recall6=plot_five_ROC(y_val_fusion,pred_fusion_val,pred_cgm_val,pred_clinical_val,pred_rf_val[:,1],pred_lr_val[:,1],pred_dt_val[:,1],y_val_CGM,y_val_clinical,save_path)
        val_data = pd.DataFrame({"y_fusion":y_val_fusion,"y_pred":pred_fusion_val,"y_clinical":y_val_clinical,"pred_clinical":pred_clinical_val,"y_rf":y_val_clinical,"pred_rf":pred_rf_val[:,1],"y_lr":y_val_clinical,"pred_lr":pred_lr_val[:,1],"y_dt":y_val_clinical,"pred_dt":pred_dt_val[:,1]})
        val_cgm = pd.DataFrame({"y_cgm":y_val_CGM,"pred_cgm":pred_cgm_val})
        auc_train,acc_train,prec_train,recall_train = bootstrap_auc(val_data,val_cgm)

        print("-----------------------------------------------------------------------Validation---------------------------------------------------------------------------")
        print("----Metric--|----------FM-----------|-----------CGM---------|------------CL---------|------------RF---------|------------LR---------|------------DT---------")
        print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("AUC",auc1,auc_train[0,0],auc_train[0,1],auc2,auc_train[5,0],auc_train[5,1],auc3,auc_train[1,0],auc_train[1,1],auc4,auc_train[2,0],auc_train[2,1],auc5,auc_train[3,0],auc_train[3,1],auc6,auc_train[4,0],auc_train[4,1]))
        print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("Accuracy",accuracy1,acc_train[0,0],acc_train[0,1],accuracy2,acc_train[5,0],acc_train[5,1],accuracy3,acc_train[1,0],acc_train[1,1],accuracy4,acc_train[2,0],acc_train[2,1],accuracy5,acc_train[3,0],acc_train[3,1],accuracy6,acc_train[4,0],acc_train[4,1]))
        #print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("Precision",precision1,prec_train[0,0],prec_train[0,1],precision2,prec_train[5,0],prec_train[5,1],precision3,prec_train[1,0],prec_train[1,1],precision4,prec_train[2,0],prec_train[2,1],precision5,prec_train[3,0],prec_train[3,1],precision6,prec_train[4,0],prec_train[4,1]))
        #print("|%10s | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) | %.4f(%.4f,%.4f) |"%("Recall",recall1,recall_train[0,0],recall_train[0,1],recall2,recall_train[5,0],recall_train[5,1],recall3,recall_train[1,0],recall_train[1,1],recall4,recall_train[2,0],recall_train[2,1],recall5,recall_train[3,0],recall_train[3,1],recall6,recall_train[4,0],recall_train[4,1],))
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------")
    
    print(0)

        

