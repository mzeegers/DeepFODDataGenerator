#UNet network training script using data generated by workflow

# This script carries out the UNet network training using the workflow-generated (experimental) training data
# The results are the networks and log files generated from the UNet code
# The code assumes that the (experimental) training to be available (see below)
# If not done beforehand, run the ReconstructAndProject script first (located in the scripts/experimental/generation folder) to create it
# The ratio between training and validation data is the same as training with MSD (needs to be run first)
# The log files from MSD training indicating which folder instances are used should be copied into the folder before running this script

# Code assumes training data to be saved in the following way
# - /data/TrainingDataExperimental/Instance001/
# -  |                            /Instance002/
# -  |                                 ...
# -  --/TrainingDataExperimentalGT/Instance001/
# -                               /Instance002/ 
# -                                    ...

# Note: The subfolders in TrainingDataExperimental and TrainingDataExperimental should have identical names

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)
#UNet code partially derived from PyTorch UNet code (https://github.com/usuyama/pytorch-unet) by Naoto Usuyama

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
import sys
import tifffile
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DataPath = '../../../../../data/' #Location of data and GT
DatasetName = 'TrainingDataExperimental' #Selected data folder and printed folder in network name
GTFolder = '' #Change if path in Datapath folder is deeper than GTName below
GTName = 'TrainingDataExperimentalGT' #Selected GT folder and printed folder in network name
targetLabels = 2 #Number of different labels in GT
NoFO = [] #CT scans without foreign objects - not used for training

#Get script arguments
NumObj = int(sys.argv[1])
Run = int(sys.argv[2])

RunString = 'Run' + str(Run)
print('Number of objects:', NumObj, ', run:', RunString) 

#Set random seeds
seed=Run
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

#Set CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Full GT path
GT = GTFolder + GTName 


#Select all instances and names of folders containing CT scan data (with few foreign objects)
flsinFolders = []
flstgFolders = []
flsinFoldersPossible = sorted(os.listdir(DataPath + DatasetName + '/'))[111:131+len(NoFO)]
flstgFoldersPossible = sorted(os.listdir(DataPath + GT + '/'))[111:131+len(NoFO)]
    
#Take the same folders as from MSD training (NOTE: the log files from the MSD training have to be copied in this folder)
with open('TrainSettingsFolderNumObj' + str(NumObj) + 'Run' + str(Run) +'.txt', 'r') as f:
    line = f.readline()
    for line in f:
        number = int(line[-3:-1]) - 1
        print(number)
        flsinFolders = flsinFolders + [flsinFoldersPossible[number]]
        flstgFolders = flstgFolders + [flstgFoldersPossible[number]]
print(flsinFolders)


# Create list of all files for training and validation set
flsin2 = [] #Input files
flstg2 = [] #Target files
for f in flsinFolders:
    flsintmp = []
    flstgtmp = []
    #Select all (image) files in CT folder
    flsintmp = flsintmp + sorted(os.listdir(DataPath + DatasetName + '/' + f + '/'))
    flstgtmp = flstgtmp + sorted(os.listdir(DataPath + GT + '/' + f + '/'))
    #Compose full path of these files
    flsin2 = flsin2 + [DataPath + DatasetName + '/' + f + '/' + s for s in flsintmp]
    flstg2 = flstg2 + [DataPath + GT + '/' + f + '/' + s for s in flstgtmp]

#Construct the training set
flsin = [] #Input files - training set
flstg = [] #Target files - training set
for j in range(0,9): #For every 10 files we select 9 for the training set
    flsin = flsin + [flsin2[i] for i in np.linspace(0,len(flsin2),1800, endpoint = False).astype(int)[j::10]]
    flstg = flstg + [flstg2[i] for i in np.linspace(0,len(flstg2),1800, endpoint = False).astype(int)[j::10]]

print("All files for training")
print(flsin)
print(flstg)
print(len(flsin))

DataListTrain = flsin
GTListTrain = flstg

#Construct the validation set (no data augmentation)
flsin = [] #Input files - validation set
flstg = [] #Target files - validation set
for j in range(9,10):
    flsin = flsin + [flsin2[i] for i in np.linspace(0,len(flsin2),1800, endpoint = False).astype(int)[j::10]]
    flstg = flstg + [flstg2[i] for i in np.linspace(0,len(flstg2),1800, endpoint = False).astype(int)[j::10]]

print("All files for validation")
print(flsin)
print(flstg)
print(len(flsin))

DataListVal = flsin
GTListVal = flstg

#Setup name to for network and log files
SettingsExt = '_' + DatasetName + GTName + RunString + 'NumObj' + str(NumObj)
print(SettingsExt)

#Helper function that separates a segmentation into multiple binary maps
def Separator(GT):
    GT2 = np.zeros((targetLabels, GT.shape[-2], GT.shape[-1]))
    for i in range(0, targetLabels):
        GT2[i,:,:][GT == i] = 1
    return GT2

#Data class
class SimDataset(Dataset):
    def __init__(self, settype = 'train'):      
        if(settype == 'train'):
            self.DataList = DataListTrain
            self.GTList = GTListTrain
        elif(settype == 'val'):
            self.DataList = DataListVal
            self.GTList = GTListVal

    def __len__(self):
        return len(self.DataList)

    def __getitem__(self, idx):
        #Load data and ground truth
        GT = Separator(tifffile.imread(self.GTList[idx])).astype(np.float32)
        Data = tifffile.imread(self.DataList[idx])
        Data = Data[np.newaxis,:]
        
        #Apply data augmentation by rotation and mirroring
        c = np.random.randint(8)
        if c==0:
            Data, GT = Data[:,::-1,:].copy(), GT[:,::-1,:].copy()
        elif c==1:
            Data, GT = Data[:,:,::-1].copy(), GT[:,:,::-1].copy()
        elif c==2:
            Data, GT = Data[:,::-1,::-1].copy(), GT[:,::-1,::-1].copy()
        elif c==3:
            Data, GT = np.rot90(Data,1,axes=(1,2)).copy(), np.rot90(GT,1,axes=(1,2)).copy()
        elif c==4:
            Data, GT = np.rot90(Data,3,axes=(1,2)).copy(), np.rot90(GT,3,axes=(1,2)).copy()
        elif c==5:
            Data, GT = np.rot90(Data,1,axes=(1,2))[:,::-1].copy(), np.rot90(GT,1,axes=(1,2))[:,::-1].copy()
        elif c==6:
            Data, GT = np.rot90(Data,3,axes=(1,2))[:,::-1].copy(), np.rot90(GT,3,axes=(1,2))[:,::-1].copy()

        return [Data, GT]


train_set = SimDataset(settype = 'train')
val_set = SimDataset(settype = 'val')

print(len(train_set), len(val_set))

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 10

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

# U-Net configuration
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )   

class WorkflowUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(1, 128)
       
        self.dconv_down2 = double_conv(128, 256)
        self.dconv_down3 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up2 = double_conv(256 + 512, 256)
        self.dconv_up1 = double_conv(256 + 128, 128)
        
        self.conv_last = nn.Conv2d(128, n_class, 1)

    def forward(self, input):

        x = input

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        x = self.dconv_down3(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

#Compute dice loss
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    #Dice loss = 1 - dice coefficient
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

#Compute loss (weighted combination of binary cross entropy loss and dice loss)
def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

#Print statistics during training
def print_metrics(metrics, epoch_samples, phase, best_loss):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)), "Best val loss: {:4f}".format(best_loss))

#Train the U-Net model
def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    log_filename = 'UNetlog_segm' + SettingsExt + '.txt'
    with open(log_filename, 'w') as f:

        for epoch in range(num_epochs):
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            since = time.time()

            # ach epoch has a training and validation phase
            for phase in ['train', 'val']:

                metrics = defaultdict(float)
                epoch_samples = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    #Set parameter gradients to zero
                    optimizer.zero_grad()

                    #Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = calc_loss(outputs, labels, metrics)

                        #Backward - only in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    #Statistics
                    epoch_samples += inputs.size(0)

                if phase == 'train':
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("LR", param_group['lr'])
                    model.train()  #Set model to training mode
                else:
                    model.eval()   #Set model to evaluate mode

                epoch_loss = metrics['loss'] / epoch_samples

                #Deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print("Saving best model...")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                    torch.save(model.state_dict(), 'UNetsegm_params' + SettingsExt + '.pth')
                    torch.save(model, 'UNetsegm_params' + SettingsExt + 'CompleteModel' + '.pth')
                    cnt = 0

                    for inputs, labels in dataloaders['val']:
                        if(cnt == 0):
                            cnt += 1
                            inputs = inputs.to(device)
                            pred = model(inputs)
                            #The loss functions include the sigmoid function.
                            pred = torch.sigmoid(pred)
                            pred = pred.data.cpu().numpy()
                            tifffile.imsave('UNetsegm_params' + SettingsExt + '_PredictedExample.tiff', pred[0,:,:,:].astype(np.float32))
                            tifffile.imsave('UNetsegm_params' + SettingsExt + '_TargetExample.tiff', labels.cpu().numpy()[0,:,:,:].astype(np.float32))

                #Show the current statistics
                print_metrics(metrics, epoch_samples, phase, best_loss)
                if phase == 'val':
                    with open(log_filename, 'a') as f:
                        f.write('Current error: {}, Best error: {}\n'.format(epoch_loss, best_loss))

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    #Get best model weights
    model.load_state_dict(best_model_wts)
    return model

#Set training parameters and train the model
num_class = targetLabels
model = WorkflowUNet(num_class).to(device)
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3000, gamma=1)
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=3000)
