
# library
# standard library
from ast import Global
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from com.utils import plot_ROC,plot_Multi_ROC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm

import model.model as mm
import random
from sklearn.linear_model import LogisticRegression,ElasticNetCV
import seaborn as sns
from loss import BCEFocalLoss
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split,KFold,RepeatedKFold
from sklearn import preprocessing
"""
    000:"no",
    001:"LGA",
    
    290:"age",
    291:"height",
    292:"BMI0",
    293:"GDMhistory",
    294:"macrohis",
    295:"parity",
    296:"sbp",
    297:"dbp",
    298:"weightplus",
    299:"insulinuse",
    300:"chol",
    301:"tg",
    302:"FPG0",
    303:"OGTT1",
    304:"hba1c0",
    305:"gbmtype",
    306:"CGMtime",
    """
if __name__ == '__main__':
    random_num = 200
    torch.manual_seed(random_num)
    # Hyper Parameters
    EPOCH =300            # train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE =10
    LR =3*0.01              # learning rate
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv('./dataset.csv',sep=',', header= None)
    
    data = np.array(data)
    data_raw = data
    #=============================================== preprocessing================================
    dim=290
    # for ind in range(data.shape[0]):
    #     data[ind,2:dim]=(data[ind,2:dim]-np.min(data[ind,2:dim]))/(np.max(data[ind,2:dim])-np.min(data[ind,2:dim]))
    # for ind in range(15):
    #     data[:,290+ind]=(data[:,290+ind]-np.min(data[:,290+ind]))/(np.max(data[:,290+ind])-np.min(data[:,290+ind]))
    for ind in range(16):
        data[:,290+ind]=preprocessing.MinMaxScaler().fit_transform(data[:,290+ind].reshape(-1,1)).reshape(-1)
    root_path = "./results/clinical"
    #-----------------------------------splitting data---------------------------------------
    data = np.concatenate((data[:,1].reshape(-1,1),data[:,0].reshape(-1,1),data[:,2:]),axis=1)
    x_train,x_test,y_train,y_test=train_test_split(data[:,1:],data[:,0],test_size=0.13,random_state=random_num)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    train_dataset = np.concatenate((y_train,x_train),axis=1)
    test_dataset = np.concatenate((y_test,x_test),axis=1) 

    x_test = test_dataset[:,:]
    y_test = test_dataset[:,0]
    p=1
    kfold=10
    kf = RepeatedKFold(n_splits=kfold,n_repeats=p,random_state=random_num)
    kf_list = list(kf.split(train_dataset))
    for index in range(p*kfold):
        print("--------%d fold----------"%(index))
        result_path = os.path.join(root_path,str(index)+"-fold")
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        #-------------------------Splitting data-------------------------
        kf_train,kf_val = kf_list[index]
        x_train = train_dataset[kf_train,:].astype(None)
        y_train = train_dataset[kf_train,0].astype(None)
        
        x_val = train_dataset[kf_val,:].astype(None)
        y_val = train_dataset[kf_val,0].astype(None)

        #============================ train dataset ===================================
        train_data = Data.TensorDataset(torch.FloatTensor(x_train),torch.Tensor(y_train))
        train_loader = Data.DataLoader(dataset=train_data,batch_size= BATCH_SIZE, shuffle = True)

        train_loader_val = Data.DataLoader(dataset=train_data,batch_size= x_train.shape[0], shuffle = False)
        #============================ validation dataset ===================================
        test_data = Data.TensorDataset(torch.FloatTensor(x_test),torch.Tensor(y_test))
        test_loader = Data.DataLoader(dataset=test_data,batch_size= x_test.shape[0], shuffle = False)
        #============================ validation dataset ===================================
        val_data = Data.TensorDataset(torch.FloatTensor(x_val),torch.Tensor(y_val))
        val_loader = Data.DataLoader(dataset=val_data,batch_size= x_val.shape[0], shuffle = False)

        mlp = mm.MLP()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)#weight_decay=0.001
        scheduler=lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.5)
        #scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=5)
        #loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
        #loss_func = nn.BCELoss()   
        loss_func = nn.BCEWithLogitsLoss()                   # the target label is not one-hotted
        
        #loss_func = BCEFocalLoss()
        # training and testing
        loss_list_train =[]
        loss_list_test=[]
        loss_list_val=[]
        auc_train = []
        auc_test = []
        auc_val = []
        acc_train = []
        acc_test = []
        acc_val = []
        precision_train = []
        precision_test = []
        precision_val = []
        recall_train = []
        recall_test = []
        recall_val = []
        AUC_best=0.60
        for epoch in range(EPOCH):
            mlp.train()
            for step, (data_batch,label) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
                output,feature= mlp(data_batch)  
                loss = loss_func(output.view(-1),label)   # cross entropy loss
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients updata network parameters
            #scheduler.step()
            # if epoch%10==0:
            #     print('Epoch: ', epoch, '| train loss: %.6f' % loss.cpu().data.numpy())
        #================================validation========================================
            mlp.eval()
            p = []
            g = []
            with torch.no_grad():
                for step,(data_batch,label) in enumerate(train_loader_val):
                    output,feature=mlp(data_batch)
                    loss_train = loss_func(output.view(-1),label)
                    output = torch.sigmoid(output) 
                    p.append(output.view(-1).tolist())
                    g.append(label.tolist())
                loss_list_train.append(loss_train.cpu().data.numpy())
                p = torch.tensor(p).view(-1)
                g = torch.tensor(g).view(-1)
                save_path = os.path.join(result_path,"mlp_roc_validation"+".png")
                AUC,accuracy,precision,recall = plot_ROC(g,p,save_path)
                auc_train.append(AUC)
                acc_train.append(accuracy)
                precision_train.append(precision)
                recall_train.append(recall)
            #================================test========================================
            pred = []
            gt = []
            with torch.no_grad():
                for step, (data_batch,label) in enumerate(test_loader): 
                    label = label
                    output,feature = mlp(data_batch)  
                    loss_test = loss_func(output.view(-1),label) 
                    output = torch.sigmoid(output) 
                    pred.append(output.view(-1).tolist())
                    gt.append(label.tolist())
                loss_list_test.append(loss_test.cpu().data.numpy())  
                pred = torch.tensor(pred).view(-1)
                gt = torch.tensor(gt).view(-1)
                AUC,accuracy,precision,recall = plot_ROC(gt,pred,save_path)
                auc_test.append(AUC)
                acc_test.append(accuracy)
                recall_test.append(recall)
                precision_test.append(precision)
            if epoch%20==0:
                print("epoch:%4d,train loss: %5f,test loss: %5f"%(epoch,loss_train,loss_test))
            if AUC > AUC_best and precision >0.1 and recall>0.1 and epoch >10:
                #AUC_best=AUC
                print("auc_Best:","AUC:",AUC,"accuracy:",accuracy,"precision:",precision,"recall:",recall)
                save_path = os.path.join(result_path,"CGM_fusion_model"+str(AUC)+"_"+str(accuracy)+"_"+str(precision)+"_"+str(recall)+"_"+str(epoch)+".pt")
                torch.save(mlp.state_dict(),save_path)
        #================================validation========================================
            ppp =[]
            ggg =[]
            mlp.eval()
            with torch.no_grad():
                for step, (data_batch,label) in enumerate(val_loader): 
                    output,feature = mlp(data_batch)
                    loss_val = loss_func(output.view(-1),label)    
                    output = torch.sigmoid(output)   
                    ppp.append(output.view(-1).tolist())
                    ggg.append(label.tolist()) 
                loss_list_val.append(loss_val.cpu().data.numpy()) 
                ppp= torch.tensor(ppp).view(-1)
                ggg= torch.tensor(ggg).view(-1)
                auc,accuracy,precision,recall = plot_ROC(ggg,ppp,save_path)
                auc_val.append(auc)
                acc_val.append(accuracy)
                precision_val.append(precision)
                recall_val.append(recall) 
        plt.figure()
        plt.plot(loss_list_train)
        plt.plot(loss_list_test)
        plt.plot(loss_list_val)
        plt.legend(["training loss","test loss","validation loss"])
        save_path = os.path.join(result_path,"loss_comparison"+".png")
        plt.savefig(save_path)
        plt.close()

        plt.figure()
        plt.plot(auc_train)
        plt.plot(auc_test)
        plt.plot(auc_val)
        plt.legend(["training auc","test auc","val auc"])
        save_path = os.path.join(result_path,"auc_comparison"+".png")
        plt.savefig(save_path)
        plt.close()

        plt.figure()
        plt.plot(acc_train)
        plt.plot(acc_test)
        plt.plot(acc_val)
        plt.legend(["training acc","test acc","val acc"])
        save_path = os.path.join(result_path,"acc_comparison"+".png")
        plt.savefig(save_path)
        plt.close()

        plt.figure()
        plt.plot(precision_train)
        plt.plot(precision_test)
        plt.plot(precision_val)
        plt.legend(["training precision","test precision","val precision"])
        save_path = os.path.join(result_path,"precision_comparison"+".png")
        plt.savefig(save_path)
        plt.close()

        plt.figure()
        plt.plot(recall_train)
        plt.plot(recall_test)
        plt.plot(recall_val)
        plt.legend(["training recall","test racall","val racall"])
        save_path = os.path.join(result_path,"recall_comparison"+".png")
        plt.savefig(save_path)
        plt.close()
    print(1)
    
