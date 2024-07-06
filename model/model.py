import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import model.resnet1d as resnet
'''
torch.save(model_object,'resnet.pth')
model=torch.load('resnet.pth')

torch.save(my_resnet.state_dict(),'my_resnet.pth')
my_resnet.load_state_dict(torch.load('my_resnet.pth'))

'''
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1),
        )
        
    def forward(self, x):
        x_gdm = x[:,290:306]
        feature  = self.mlp(x_gdm)
        output=feature
        return output,feature
class MLP_fusion(nn.Module):
    def __init__(self):
        super(MLP_fusion, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=8,out_channels=16,kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mlp = nn.Sequential(
         
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(11,16),
            nn.ReLU(),
        )
        self.fc=nn.Sequential(
            nn.Linear(32,1),
        )
        self.classifier = nn.Sigmoid()

    def forward(self, x):
        x_cgm = x[:,2:290]
        x_clinical = x[:,290:301]
        x_cgm = x_cgm.reshape(-1,1,288)
        feature_1  = self.cnn(x_cgm)
        
        featuremap = feature_1
        feature_1  = self.mlp(feature_1.view(-1,feature_1.shape[1]))
        feature_2  = self.encoder(x_clinical)
        feature = torch.cat([feature_1,feature_2],dim=1)
        feature=self.fc(feature)
        output = feature
        #output = self.classifier(feature)
        return output,featuremap

        
class MLP_CGM(nn.Module):
    def __init__(self):
        super(MLP_CGM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=8,out_channels=16,kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),

        )
        self.mlp = nn.Sequential(
            nn.Linear(32,16),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(16,1),
        )

    def forward(self, x):
        x_cgm = x[:,2:290]
        # plt.plot(x_cgm)
        # if x[0,1]==0:
        #     plt.savefig(os.path.join("/home/user/Code/GMD/bloodglucose_two_groups/results/CGM_total/0",str(np.array(x[0,0]))+".jpg"))
        # else:
        #     plt.savefig(os.path.join("/home/user/Code/GMD/bloodglucose_two_groups/results/CGM_total/1",str(np.array(x[0,0]))+".jpg"))
        # plt.close()
        x_cgm = x_cgm.reshape(-1,1,288)
        feature  = self.cnn(x_cgm)
        feature  = self.mlp(feature.view(-1,feature.shape[1]))
        output = feature
        return output,feature
class CNN_MLP(nn.Module):
    def __init__(self):
        super(CNN_MLP, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=8,out_channels=16,kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.mlp = nn.Sequential(
         
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
        )
        self.classifier = nn.Sigmoid()

    def forward(self, x):
        x_cgm = x[:,:288]
        x_cgm = x_cgm.reshape(-1,1,288)
        x_cgm = self.cnn(x_cgm)
        x_cgm = x_cgm.view(x_cgm.shape[0],x_cgm.shape[1])
        feature  = self.mlp(x_cgm)
        output=feature
        #output = self.classifier(feature)
        return output,feature
class MLP_fusion_v1(nn.Module):
    def __init__(self):
        super(MLP_fusion_v1, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3),
            #nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=8,out_channels=16,kernel_size=3),
            #nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mlp = nn.Sequential(
         
            nn.Linear(288,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(10,16),
            nn.ReLU(),
        )

        self.fc=nn.Sequential(
            nn.Linear(32,1),
        )
        self.classifier = nn.Sigmoid()
        self.embedding = nn.Embedding(1372,1)

    def forward(self, x):

        x_cgm = x[:,2:290]
        x_clinical = x[:,290:300]
        x_cgm = x_cgm.reshape(-1,1,288)
        feature_1  = self.cnn(x_cgm)
        #feature_1  = self.mlp(x_cgm)
        feature_2  = self.encoder(x_clinical)
        feature = torch.cat([feature_1.view(-1,feature_1.shape[1]),feature_2],dim=1)
        feature=self.fc(feature)
        output = feature
        #output = self.classifier(feature)
        return output,feature_1    
