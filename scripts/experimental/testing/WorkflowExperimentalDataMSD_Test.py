#Testing scripts for trained MSD networks on workflow generated data

# This script carries out the testing of MSD networks on the workflow-generated (experimental) testing data
# The results are the csv files in the results/quantitative/ folder

# The code assumes that the MSD networks are rained and available in the scripts/experimental/training/MSD/ folders
# Any untrained or undesired network configuration can be removed below.

# CT data for testing are the ones with instance numbers 66-111 and 9, 14, 26, 38, 50, 59 (not containing foreign objects)
# Code assumes testing data to be saved in the following way
# - /data/TrainingDataExperimental/    ...
# -  |                            /Instance066/
# -  |                            /Instance067/
# -  |                                 ...
# -  --/TrainingDataExperimentalGT/    ...
# -                               /Instance066/
# -                               /Instance067/ 
# -                                    ...

# Note: The subfolders in TrainingDataExperimental and TrainingDataExperimental should have identical names

#Author,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import csv
import cv2
import glob
import matplotlib.pyplot as plt
import msdnet
import numpy as np
import os
from pathlib import Path
import sklearn.metrics as m
import time
import tifffile

#Path to generated quantitative to save
pathToQuant = '/../../../results/experimental/quantitative'

#Function to produce a labelled image from an image of components
def LabelComponents(img):
   
    num_labels, labels = cv2.connectedComponents(img)
    print("Num_labels", num_labels)
                
    #Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    #Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    #Set background label to black
    labeled_img[label_hue==0] = 0
                
    labeled_img2 = np.zeros((labeled_img.shape[0],labeled_img.shape[1]))
    for count, value in enumerate(np.unique(labeled_img.reshape(labeled_img.shape[0]*labeled_img.shape[1],labeled_img.shape[2]), axis=0)):
        labeled_img2[(labeled_img[:,:,0] == value[0]) & (labeled_img[:,:,1] == value[1]) & (labeled_img[:,:,2] == value[2])] = count
    return labeled_img2


#Load data files
RootDataPath = '../../../data/' #'/export/scratch3/zeegers/AutomatedFODProjectExperimentalData/'
FullDataPath = RootDataPath + 'TrainingDataExperimental/'
FullGTPath = RootDataPath + 'TrainingDataExperimentalGT/'

flsin = []
flstg = []
FileListData = sorted(os.listdir(FullDataPath))
FileListGT = sorted(os.listdir(FullGTPath))
flsinFolders = FileListData[66:112] + [FileListData[9]] + [FileListData[14]] + [FileListData[26]] + [FileListData[38]] + [FileListData[50]] + [FileListData[59]]
flstgFolders = FileListGT[66:112] + [FileListGT[9]] + [FileListGT[14]] + [FileListGT[26]] + [FileListGT[38]] + [FileListGT[50]] + [FileListGT[59]]
for f in flsinFolders:
    flsintmp = sorted(os.listdir(FullDataPath + f + '/'))[900::450] #We take two angles perpendicular to each other
    flstgtmp = sorted(os.listdir(FullGTPath + f + '/'))[900::450]

    flsin = flsin + [FullDataPath + f + '/' + s for s in flsintmp]
    flstg = flstg + [FullGTPath + f + '/' + s for s in flstgtmp]


Prefix = '../training/'

curPath = os.path.dirname(os.path.abspath(__file__))


def computeNetworkResults(NetworkPath, Runs, NumObjs, flsin):

    SegAcc = np.zeros((len(NumObjs),len(Runs)))
    F1ScoreAcc = np.zeros((len(NumObjs),len(Runs)))
    Specialmetric = np.zeros((len(NumObjs), len(Runs)))
    FPmetric = np.zeros((len(NumObjs), len(Runs)))

    RunIndex = 0
    for Run in Runs:

        RunString = 'Run' + str(Run)

        cnt = 0
        
        for NumObj in NumObjs:
            print("Run", Run, "NumObj", NumObj)
            
            #Load the corresponding trained network
            subname = 'TrainingDataExperimentalTrainingDataExperimentalGT' + RunString + 'NumObj' + str(NumObj)
            FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10.h5'
            n = msdnet.network.SegmentationMSDNet.from_file(FullNetworkPath)

            #Initialize
            cntSpecialmetric = 0
            cntSpecialmetricTotal = 0
            cntFPmetricTotal = 0
            cntFPmetric = 0
            TotalPixErr = 0
            TotalPixErrNorm = 0
            TotalTPrate = 0 
            TotalF1Score = 0
             
            for i in range(len(flsin)):

                #Load files and compute network result
                d = msdnet.data.ImageFileDataPoint(flsin[i])
                output = n.forward(d.input)
                segment = np.argmax(output,0)
                target = tifffile.imread(flstg[i])

                pixerr = np.count_nonzero(segment != target)
                pixerrnorm = np.count_nonzero(segment != target)/target.size

                #First segmentation metric (F1 score)
                F1Score = m.f1_score(target.flatten(), segment.flatten())

                #Second segmentaton metric (average class accuracy)
                TPrate = 0
                for j in range(0,len(np.unique(target))):
                    TPrate += np.count_nonzero(np.logical_and((segment == j),(target == j)))/np.count_nonzero(target == j)
                TPrate = TPrate/len(np.unique(target))
                print(i)
                
                #Third segmentation metric (object based detection rate)
                targetLabeled = LabelComponents(target)
                cntSpecialmetricTotal += len(np.unique(targetLabeled))-1
                if(len(np.unique(targetLabeled)) == 1):
                    print("No foreign object in ground truth")
                    
                for val in np.unique(targetLabeled):
                    if(val > 0):
                        if(np.count_nonzero(targetLabeled == val) > 8):
                            targetLabeledTemp = targetLabeled.copy()
                            targetLabeledTemp[targetLabeled != val] = 0
                            targetLabeledTemp[targetLabeled == val] = 1
                            if( np.count_nonzero(np.logical_and((segment == 1),(targetLabeledTemp == 1)))/np.count_nonzero((targetLabeledTemp == 1)) > 0.3):
                                cntSpecialmetric += 1
                                print("Foreign object detected: Yes")
                        else:
                            cntSpecialmetricTotal -= 1        
                                
                print("Current found", cntSpecialmetric, ", Current ", cntSpecialmetricTotal)

                #Fourth segmentation metric (object based false positive detection rate)
                targetLabeledOutput = LabelComponents(segment.astype(np.uint8))

                cntFPmetricTotal += len(np.unique(targetLabeledOutput))-1
                for val in np.unique(targetLabeledOutput):
                    if(val > 0):
                        if(np.count_nonzero(targetLabeledOutput == val) > 8):
                            targetLabeledOutputTemp = targetLabeledOutput.copy()
                            targetLabeledOutputTemp[targetLabeledOutput != val] = 0
                            targetLabeledOutputTemp[targetLabeledOutput == val] = 1
                            if( np.count_nonzero(np.logical_and((target == 1),(targetLabeledOutputTemp == 1)))/np.count_nonzero((targetLabeledOutputTemp == 1)) <= 0.3):  
                                cntFPmetric += 1
                                print("False positive detected, label:", val)
                        else:
                            cntFPmetricTotal -= 1   

                TotalPixErr += pixerr
                TotalPixErrNorm += pixerrnorm
                TotalTPrate += TPrate
                TotalF1Score += F1Score
            TotalPixErr = TotalPixErr/len(flsin)
            TotalPixErrNorm = TotalPixErrNorm/len(flsin)
            TotalTPrate = TotalTPrate/len(flsin)
            TotalF1Score = TotalF1Score/len(flsin)

            print('Results:')
            print('Overall accuracy', (1-TotalPixErrNorm)*100)
            print('Average class accuracy', TotalTPrate*100)
            print('F1 score', TotalF1Score*100)
            print('cntSpecialmetric', cntSpecialmetric, 'total', cntSpecialmetricTotal, 'metric', cntSpecialmetric/float(cntSpecialmetricTotal))
            if(float(cntFPmetricTotal) == 0):
                print('False positives', cntFPmetric, 'total', cntFPmetricTotal, 'metric', 0)
            else:
                print('False positives', cntFPmetric, 'total', cntFPmetricTotal, 'metric', cntFPmetric/float(cntFPmetricTotal))

            SegAcc[cnt,RunIndex] += TotalTPrate*100
            F1ScoreAcc[cnt,RunIndex] += TotalF1Score*100
            Specialmetric[cnt,RunIndex] += cntSpecialmetric/float(cntSpecialmetricTotal)*100
            if(float(cntFPmetricTotal) == 0):
                FPmetric[cnt,RunIndex] += 0
            else:
                FPmetric[cnt,RunIndex] += cntFPmetric/float(cntFPmetricTotal)*100
            cnt += 1

        RunIndex += 1
       
    return SegAcc, F1ScoreAcc, Specialmetric, FPmetric


######################
### FEW FOs        ###
######################

NumObjs = [1,2,3,4,5,7,10,15,20,30,40,50,60]
Runs = [0,1,2,3,4]
NetworkPath = Prefix + '/MSD/FewFOs/'

SegAcc, F1ScoreAcc, Specialmetric, FPmetric = computeNetworkResults(NetworkPath, Runs, NumObjs, flsin)

#Compute means and standard deviations of all measures
SegAccAvg = np.mean(SegAcc, axis = 1)
F1ScoreAccAvg = np.mean(F1ScoreAcc, axis = 1)
SpecialmetricAvg = np.mean(Specialmetric, axis = 1)
FPmetricAvg = np.mean(FPmetric, axis = 1)
SegAccStd = np.std(SegAcc, axis = 1)
F1ScoreAccStd = np.std(F1ScoreAcc, axis = 1)
SpecialmetricStd = np.std(Specialmetric, axis = 1)
FPmetricStd = np.std(FPmetric, axis = 1)

print(SegAcc)
print(F1ScoreAcc)
print(Specialmetric)
print(FPmetric)


######################
### MANY FOs       ###
######################

NumObjsManyFO = [1,2,3,4,5,7,10,15,20]
Runs = [0,1,2,3,4]
NetworkPath = Prefix + '/MSD/ManyFOs/'

SegAccManyFO, F1ScoreAccManyFO, SpecialmetricManyFO, FPmetricManyFO = computeNetworkResults(NetworkPath, Runs, NumObjsManyFO, flsin)

#Compute means and standard deviations of all measures
SegAccAvgManyFO = np.mean(SegAccManyFO, axis = 1)
F1ScoreAccAvgManyFO = np.mean(F1ScoreAccManyFO, axis = 1)
SpecialmetricAvgManyFO = np.mean(SpecialmetricManyFO, axis = 1)
FPmetricAvgManyFO = np.mean(FPmetricManyFO, axis = 1)
SegAccStdManyFO = np.std(SegAccManyFO, axis = 1)
F1ScoreAccStdManyFO = np.std(F1ScoreAccManyFO, axis = 1)
SpecialmetricStdManyFO = np.std(SpecialmetricManyFO, axis = 1)
FPmetricStdManyFO = np.std(FPmetricManyFO, axis = 1)

print(SegAccManyFO)
print(F1ScoreAccManyFO)
print(SpecialmetricManyFO)
print(FPmetricManyFO)


######################
### MIXED          ###
######################

NumObjsMix = [1,2,3,4,5,7,10,15,20,30,40]
Runs = [0,1,2,3,4]
NetworkPath = Prefix + '/MSD/MixedFOs/'

SegAccMix, F1ScoreAccMix, SpecialmetricMix, FPmetricMix = computeNetworkResults(NetworkPath, Runs, NumbObjsMix, flsin)

#Compute means and standard deviations of all measures
SegAccAvgMix = np.mean(SegAccMix, axis = 1)
F1ScoreAccAvgMix = np.mean(F1ScoreAccMix, axis = 1)
SpecialmetricAvgMix = np.mean(SpecialmetricMix, axis = 1)
FPmetricAvgMix = np.mean(FPmetricMix, axis = 1)
SegAccStdMix = np.std(SegAccMix, axis = 1)
F1ScoreAccStdMix = np.std(F1ScoreAccMix, axis = 1)
SpecialmetricStdMix = np.std(SpecialmetricMix, axis = 1)
FPmetricStdMix = np.std(FPmetricMix, axis = 1)

print(SegAccMix)
print(F1ScoreAccMix)
print(SpecialmetricMix)
print(FPmetricMix)


######################
### ONEPROJ        ###
######################

NumObjsOneProj = [2,3,4,5,7,10,20,30,40,50,60]
Runs = [0,1,2,3,4]
NetworkPath = Prefix + '/MSD/FewFOsOneRadiograph/'

SegAccOneProj, F1ScoreAccOneProj, SpecialmetricOneProj, FPmetricOneProj = computeNetworkResults(NetworkPath, Runs, NumObjsOneProj, flsin)

#Compute means and standard deviations of all measures
SegAccAvgOneProj = np.mean(SegAccOneProj, axis = 1)
F1ScoreAccAvgOneProj = np.mean(F1ScoreAccOneProj, axis = 1)
SpecialmetricAvgOneProj = np.mean(SpecialmetricOneProj, axis = 1)
FPmetricAvgOneProj = np.mean(FPmetricOneProj, axis = 1)
SegAccStdOneProj = np.std(SegAccOneProj, axis = 1)
F1ScoreAccStdOneProj = np.std(F1ScoreAccOneProj, axis = 1)
SpecialmetricStdOneProj = np.std(SpecialmetricOneProj, axis = 1)
FPmetricStdOneProj = np.std(FPmetricOneProj, axis = 1)

print(SegAccOneProj)
print(F1ScoreAccAvgOneProj)
print(SpecialmetricOneProj)
print(FPmetricOneProj)


# Write raw results to csv file (row are number of objects, columns are runs)
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_AvgClassAcc.csv', SegAcc, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_F1ScoreAcc.csv', F1ScoreAcc, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_DetAcc.csv', Specialmetric, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_FPrate.csv', FPmetric, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_ManyFO_AvgClassAcc.csv', SegAccManyFO, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_ManyFO_F1ScoreAcc.csv', F1ScoreAccManyFO, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_ManyFO_DetAcc.csv', SpecialmetricManyFO, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_ManyFO_FPrate.csv', FPmetricManyFO, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_Mixed_AvgClassAcc.csv', SegAccMix, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_Mixed_F1ScoreAcc.csv', F1ScoreAccMix, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_Mixed_DetAcc.csv', SpecialmetricMix, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_Mixed_FPrate.csv', FPmetricMix, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_OneProj_AvgClassAcc.csv', SegAccOneProj, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_OneProj_F1ScoreAcc.csv', F1ScoreAccOneProj, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_OneProj_DetAcc.csv', SpecialmetricOneProj, delimiter=' ', fmt='%f')
np.savetxt(curPath + pathToQuant + '/RawResults_Paper_MSD_OneProj_FPrate.csv', FPmetricOneProj, delimiter=' ', fmt='%f')
