import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
def plot_ROC(labels, preds, savepath):

    fpr1,tpr1,threshold1=metrics.roc_curve(labels,preds,pos_label=1)
    roc_auc=metrics.auc(fpr1,tpr1)
    preds[preds>=0.5]=1
    preds[preds<0.5]=0
    accuracy = metrics.accuracy_score(labels,preds)
    if torch.sum(preds)==0:
        precision=0
    else:
        precision = metrics.precision_score(labels,preds)
    recall = metrics.recall_score(labels,preds)
    lw=2
    return roc_auc,accuracy,precision,recall

def plot_Multi_ROC(label1, pred1, pred2, pred3, pred4,pred5,pred6,savepath):

    fpr1,tpr1,threshold1=metrics.roc_curve(label1,pred1)
    roc_auc1=metrics.auc(fpr1,tpr1)

    fpr2,tpr2,threshold1=metrics.roc_curve(label1,pred2)
    roc_auc2=metrics.auc(fpr2,tpr2)

    fpr3,tpr3,threshold1=metrics.roc_curve(label1,pred3)
    roc_auc3=metrics.auc(fpr3,tpr3)

    fpr4,tpr4,threshold1=metrics.roc_curve(label1,pred4)
    roc_auc4=metrics.auc(fpr4,tpr4)
    fpr5,tpr5,threshold1=metrics.roc_curve(label1,pred5)
    roc_auc5=metrics.auc(fpr5,tpr5)
    fpr6,tpr6,threshold1=metrics.roc_curve(label1,pred6)
    roc_auc6=metrics.auc(fpr6,tpr6)
    plt.figure()
    lw=2
    plt.figure(figsize=(10,10))
    plt.plot(fpr1,tpr1,color='cyan',lw=lw,label='Ours model(Clinical+CGM) AUC=%0.3F'%roc_auc1)
    plt.plot(fpr2,tpr2,color='blue',lw=lw,label='Random forest AUC=%0.3F'%roc_auc2)
    plt.plot(fpr3,tpr3,color='red',lw=lw,label='Support vector machine AUC=%0.3F'%roc_auc3)
    plt.plot(fpr4,tpr4,color='magenta',lw=lw,label='ElasticNet AUC=%0.3F'%roc_auc4)
    plt.plot(fpr5,tpr5,color='pink',lw=lw,label='Logistic Regressor AUC=%0.3F'%roc_auc5)
    plt.plot(fpr6,tpr6,color='yellow',lw=lw,label='Ours model(CGM) AUC=%0.3F'%roc_auc6)

    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc="lower right")
    #plt.legend(["No SUV_max","SUV_max"],loc="lower right")
    plt.savefig(savepath)
    plt.close()#
    return roc_auc1,roc_auc2,roc_auc3,roc_auc4,roc_auc5,roc_auc6

def plot_box(data):
    columns_name=["Age","BMI0","sbp","dbp","weightplus","chol","KFYDS","THXTDB"]
    
    cnt=0
    plt.figure()
    for index in [290,291,295,296,297,299,230,231]:
        plt.subplot(2,4,cnt+1)
        plt.title(columns_name[cnt])
        lga_1 = data[data[:,1]==0][:,index]
        lga_2 = data[data[:,1]==1][:,index]
        cnt+=1
        plt.boxplot([lga_1,lga_2],labels=["LGA-0","LGA-1"])
        plt.subplots_adjust(left=None,wspace=0.5,hspace=0.5)
    plt.close()
def plot_five_ROC(label1, pred1, pred2, pred3, pred4,pred5,pred6,label2,label3,savepath):
    thre=0.5
    pred =pred1.clone()
    fpr1,tpr1,threshold1=metrics.roc_curve(label1,pred)
    roc_auc1=metrics.auc(fpr1,tpr1)
    pred[pred>=thre]=1
    pred[pred<thre]=0
    accuracy1 = metrics.accuracy_score(label1,pred)
    precision1 = metrics.precision_score(label1,pred)
    recall1 = metrics.recall_score(label1,pred)

    pred =pred2.clone()
    fpr2,tpr2,threshold1=metrics.roc_curve(label2,pred)
    roc_auc2=metrics.auc(fpr2,tpr2)
    pred[pred>=thre]=1
    pred[pred<thre]=0
    accuracy2 = metrics.accuracy_score(label2,pred)
    precision2 = metrics.precision_score(label2,pred)
    recall2 = metrics.recall_score(label2,pred)

    pred = pred3.clone()
    fpr3,tpr3,threshold1=metrics.roc_curve(label3,pred)
    roc_auc3=metrics.auc(fpr3,tpr3)
    pred[pred>=thre]=1
    pred[pred<thre]=0
    accuracy3 = metrics.accuracy_score(label3,pred)
    precision3 = metrics.precision_score(label3,pred)
    recall3 = metrics.recall_score(label3,pred)

    pred = copy.copy(pred4)
    fpr4,tpr4,threshold1=metrics.roc_curve(label1,pred)
    roc_auc4=metrics.auc(fpr4,tpr4)
    pred[pred>=thre]=1
    pred[pred<thre]=0
    accuracy4 = metrics.accuracy_score(label1,pred)
    precision4 = metrics.precision_score(label1,pred)
    recall4 = metrics.recall_score(label1,pred)

    pred = copy.copy(pred5)
    fpr5,tpr5,threshold1=metrics.roc_curve(label1,pred)
    roc_auc5=metrics.auc(fpr5,tpr5)
    pred[pred>=thre]=1
    pred[pred<thre]=0
    accuracy5 = metrics.accuracy_score(label1,pred)
    precision5 = metrics.precision_score(label1,pred)
    recall5 = metrics.recall_score(label1,pred)

    pred = copy.copy(pred6)
    fpr6,tpr6,threshold1=metrics.roc_curve(label1,pred)
    roc_auc6=metrics.auc(fpr6,tpr6)
    pred[pred>=thre]=1
    pred[pred<thre]=0
    accuracy6 = metrics.accuracy_score(label1,pred)
    precision6 = metrics.precision_score(label1,pred)
    recall6 = metrics.recall_score(label1,pred)

    plt.figure()
    lw=2
    plt.figure(figsize=(10,10))
    plt.plot(fpr1,tpr1,color='deeppink',linestyle="-",lw=lw,label='Fusion model') #: AUC=%0.3F'%roc_auc1
    plt.plot(fpr2,tpr2,color='mediumorchid',linestyle="-",lw=lw,label='CNN-based model')#: AUC=%0.3F%roc_auc2'
    plt.plot(fpr3,tpr3,color='slateblue',linestyle="-",lw=lw,label='MLP-based model')#: AUC=%0.3F'%roc_auc3
    plt.plot(fpr4,tpr4,color='orange',linestyle="-",lw=lw,label='Random Forest model')#: AUC=%0.3F%roc_auc4
    plt.plot(fpr5,tpr5,color='forestgreen',linestyle="-",lw=lw,label='Logistic Regressor model')#: AUC=%0.3F'%roc_auc5
    plt.plot(fpr6,tpr6,color='deepskyblue',linestyle="-",lw=lw,label='Decision Tree model')#: AUC=%0.3F'%roc_auc6

    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    font = {'family':'Times New Roman',
             'weight':'normal',
             'size':15}
    plt.xticks(fontproperties = 'Times New Roman',size=15)
    plt.yticks(fontproperties = 'Times New Roman',size=15)
    plt.xlabel('1-Specificity (FPR)',font)
    plt.ylabel('Sensitivity (TPR)',font)
    plt.legend(loc="lower right",prop = font)
    #plt.legend(["No SUV_max","SUV_max"],loc="lower right")
    plt.savefig(savepath,dpi=300)
    plt.close()#
    return roc_auc1,roc_auc2,roc_auc3,roc_auc4,roc_auc5,roc_auc6,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,precision1,precision2,precision3,precision4,precision5,precision6,recall1,recall2,recall3,recall4,recall5,recall6