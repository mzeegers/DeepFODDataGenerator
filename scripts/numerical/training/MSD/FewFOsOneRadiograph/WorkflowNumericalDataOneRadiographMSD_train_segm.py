#MSD network training script using data generated by workflow from simulated CT data

# This script carries out the MSD network training using the workflow-generated (numerical) training data with one radiograph per object
# The results are the networks and log files generated from the MSD code
# The code assumes that the (numerical) training to be available (see below)
# If not done beforehand, run the PhantomGeneratorTrainandTest, PhantomProjectorTrainandTest and SpectralDataGenerator script first (located in the scripts/numerical/generation folder) to create it
# The ratio between training and validation data is hardcoded to 9:1 (but can relatively easily be changed)

# Code assumes training data to be saved in the following way
# - /data/Numerical/ProjectionDataTrain/Instance001/
# -           |                        /Instance002/
# -           |                             ...
# -           --/GTProjectionsPerfectTrain/Instance001/
# -                                       /Instance002/ 
# -                                             ...

# Note: The subfolders in ProjectionDataTrain and GTProjectionsPerfectTrain should have identical names


#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)
#MSD code derived from the original MSD code (https://github.com/dmpelt/msdnet) by Daniël Pelt


import msdnet
import msdnet.operations
import numpy as np
import os
from pathlib import Path
import random
import sys
import tifffile

layers = 100
dilations = 10
DataPath = '../../../../../data/Numerical/' #Location of data and GT
Dataset = 'ProjectionDataTrain' #Selected data folder and printed folder in network name
GTFolder = '' #Change if path in Datapath folder is deeper than GTName below
GTName = 'GTProjectionsPerfectTrain' #Selected GT folder and printed folder in network name
targetLabels = 2 #Number of different labels in GT


#Get script arguments
NumObj = int(sys.argv[1])
Run = int(sys.argv[2])

RunString = 'Run' + str(Run)
print('Number of objects:', NumObj, ', run:', RunString) 

#Set random seeds
np.random.seed(Run)
random.seed(Run)

#Set number of CPU
msdnet.operations.setthreads(4)

#Full GT path
GT = GTFolder + GTName 

#Set dilations: repeatedly increasing from 1 to dilations value
dil = msdnet.dilations.IncrementDilations(dilations)

#Create main network object for segmentation 1 input channel
n = msdnet.network.SegmentationMSDNet(layers, dil, 1, targetLabels, gpu = True)

#Initialize network parameters
n.initialize()


#Select all instances and names of folders containing the simulated scan data
flsinFolders = []
flstgFolders = []
flsinFoldersPossible = sorted([d for d in os.listdir(DataPath + Dataset + '/') if os.path.isdir(os.path.join(DataPath + Dataset + '/', d))])
flstgFoldersPossible = sorted([d for d in os.listdir(DataPath + GT + '/') if os.path.isdir(os.path.join(DataPath + GT + '/', d))])
PossibleIntegers = list(range(0,len(flsinFoldersPossible)))

#Randomly select indices and corresponding CT scan folders from the remaining possibilities
for i in range(0,NumObj):
    index = random.choice(PossibleIntegers)
    PossibleIntegers.remove(index)
    flsinFolders = flsinFolders + [flsinFoldersPossible[index]]
    flstgFolders = flstgFolders + [flstgFoldersPossible[index]]

#Log the chosen folders
with open('TrainSettingsFolderNumObj' + str(NumObj) + 'Run' + str(Run) +'.txt', 'w') as f:
    f.write("Folders\n") 
    for item in flsinFolders:
        f.write("%s\n" % item)
    f.close()
print(flsinFolders)


#Construct the training set
flsin = [] #Input files - training set
flstg = [] #Target files - training set
for f in flsinFolders[0:int(np.floor(NumObj*9.0/10.0))]:
    flsintmp = []
    flstgtmp = []
    #Select all (image) files in CT folder
    flsintmp = flsintmp + sorted(os.listdir(DataPath + Dataset + '/' + f + '/'))
    flstgtmp = flstgtmp + sorted(os.listdir(DataPath + GT + '/' + f + '/'))
    #Randomly select an angle and the corresponding file
    randind = np.random.randint(0,len(flsintmp))
    flsintmp = [flsintmp[randind]]
    flstgtmp = [flstgtmp[randind]]
    #Compose full path of these files
    flsin = flsin + [DataPath + Dataset + '/' + f + '/' + s for s in flsintmp]
    flstg = flstg + [DataPath + GT + '/' + f + '/' + s for s in flstgtmp]

#Log the chosen angles and corresponding files
with open('TrainSettingsFilesTrainingNumObj' + str(NumObj) + 'Run' + str(Run) +'.txt', 'w') as f:
    f.write("Folders\n") 
    for item in flsin:
        f.write("%s\n" % item)
    f.close()

print("All files for training")
print(flsin)
print(flstg)
print(len(flsin))

#Create list of datapoints (i.e. input/target pairs) for the training set
dats = []
for i in range(len(flsin)):
    print(i)
    #Create datapoint with file names
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    #Convert datapoint to one-hot
    d_oh = msdnet.data.OneHotDataPoint(d, range(0, targetLabels))
    #Augment data by rotating and flipping
    d_augm = msdnet.data.RotateAndFlipDataPoint(d_oh)
    #Add augmented datapoint to list
    dats.append(d_augm)

#Normalize input and output of network to zero mean and unit variance using training data images
n.normalizeinout(dats)

#Use image batches of a single image
bprov = msdnet.data.BatchProvider(dats,10)

#Construct the validation set (no data augmentation)
flsin = [] #Input files - validation set
flstg = [] #Target files - validation set
for f in flsinFolders[int(np.floor(NumObj*9.0/10.0)):NumObj]:
    flsintmp = []
    flstgtmp = []
    #Select all (image) files in CT folder
    flsintmp = flsintmp + sorted(os.listdir(DataPath + Dataset + '/' + f + '/'))
    flstgtmp = flstgtmp + sorted(os.listdir(DataPath + GT + '/' + f + '/'))
    #Randomly select an angle and the corresponding file
    randind = np.random.randint(0,len(flsintmp))
    flsintmp = [flsintmp[randind]]
    flstgtmp = [flstgtmp[randind]]
    #Compose full path of these files
    flsin = flsin + [DataPath + Dataset + '/' + f + '/' + s for s in flsintmp]
    flstg = flstg + [DataPath + GT + '/' + f + '/' + s for s in flstgtmp]

with open('TrainSettingsFilesValidationNumObj' + str(NumObj) + 'Run' + str(Run) +'.txt', 'w') as f:
    f.write("Folders\n") 
    for item in flsin:
        f.write("%s\n" % item)
    f.close()

print("All files for validation")
print(flsin)
print(flstg)
print(len(flsin))

#Create list of datapoints for the validation set
datsv = []
for i in range(len(flsin)):
    print(i)
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    d_oh = msdnet.data.OneHotDataPoint(d, range(0, targetLabels))
    datsv.append(d_oh)

#Validate with Mean-Squared Error
val = msdnet.validate.MSEValidation(datsv)

#Use ADAM training algorithms
t = msdnet.train.AdamAlgorithm(n)

#Setup name to for network and log files
SettingsExt = '_' + Dataset + GTName + RunString + 'NumObj' + str(NumObj) + '_layers' + str(layers) + 'dil' + str(dilations)
print(SettingsExt)

#Log error metrics to console
consolelog = msdnet.loggers.ConsoleLogger()
#Log error metrics to file
filelog = msdnet.loggers.FileLogger('log_segm' + SettingsExt + '.txt')
#Log typical, worst, and best images to image files
imagelog = msdnet.loggers.ImageLabelLogger('log_segm' + SettingsExt , onlyifbetter=True, imsize = 500)

#Log images for each channel
singlechannellogs = []
outfolder = Path('Singlechannellogs' + SettingsExt)
outfolder.mkdir(exist_ok=True)
for i in range(0,targetLabels):
    singlechannellogs.append(msdnet.loggers.ImageLogger(str(outfolder) + '/log_segm' + SettingsExt + '_singlechannel_' + str(i), chan_out=i, onlyifbetter=True, imsize = 500))

#Train network until program is stopped manually or given time runs out
print("Training starting...")
msdnet.train.train(n, t, val, bprov, 'segm_params' + SettingsExt + '.h5', loggers=[consolelog,filelog,imagelog] + singlechannellogs, val_every=len(datsv)//10, progress = True)
