#Ploting script for results from MSD and UNet networks tested on workflow generated data (many projection vs. one projection per scan)

# This script plots the results of MSD and UNet networks on the workflow-generated (experimental) testing data
# The plots indicate the difference in results when all training examples are used in a CT rather than only one annotated example
# The results are the plots files in the results/plots/ folder

# The code assumes that the csv files with MSD and UNet network results on training data are available in the results/quantitative/ folder

#Author,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

curPath = os.path.dirname(os.path.abspath(__file__))
pathToQuant = '/../../../results/experimental/quantitative/'
pathToPlots = '/../../../results/experimental/plots/'

#Load relevant files
SegAccUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_AvgClassAcc.csv', delimiter=' ')
SpecialmetricUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_DetAcc.csv', delimiter=' ')
FPmetricUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_FPrate.csv', delimiter=' ')
F1ScoreUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_F1ScoreAcc.csv', delimiter=' ')
SegAccOneProjUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_OneProj_AvgClassAcc.csv', delimiter=' ')
SpecialmetricOneProjUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_OneProj_DetAcc.csv', delimiter=' ')
FPmetricOneProjUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_OneProj_FPrate.csv', delimiter=' ')
F1ScoreOneProjUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_OneProj_F1ScoreAcc.csv', delimiter=' ')

SegAccMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_AvgClassAcc.csv', delimiter=' ')
SpecialmetricMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_DetAcc.csv', delimiter=' ')
FPmetricMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_FPrate.csv', delimiter=' ')
F1ScoreMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_F1ScoreAcc.csv', delimiter=' ')
SegAccOneProjMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_OneProj_AvgClassAcc.csv', delimiter=' ')
SpecialmetricOneProjMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_OneProj_DetAcc.csv', delimiter=' ')
FPmetricOneProjMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_OneProj_FPrate.csv', delimiter=' ')
F1ScoreOneProjMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_OneProj_F1ScoreAcc.csv', delimiter=' ')

Runs = SegAccMSD.shape[1]

##Additional data
NumObjsOneProj = [2,3,4,5,7,10,15,20,30,40,50,60]
NumObjs = [1,2,3,4,5,7,10,15,20,30,40,50,60]

#Compute averages and stds
SegAccAvgUNet = np.mean(SegAccUNet, axis = 1)
SpecialmetricAvgUNet = np.mean(SpecialmetricUNet, axis = 1)
FPmetricAvgUNet = np.mean(FPmetricUNet, axis = 1)
F1ScoreAvgUNet = np.mean(F1ScoreUNet, axis = 1)
SegAccStdUNet = np.std(SegAccUNet, axis = 1)
SpecialmetricStdUNet = np.std(SpecialmetricUNet, axis = 1)
FPmetricStdUNet = np.std(FPmetricUNet, axis = 1)
F1ScoreStdUNet = np.std(F1ScoreUNet, axis = 1)

SegAccAvgOneProjUNet = np.mean(SegAccOneProjUNet, axis = 1)
SpecialmetricAvgOneProjUNet = np.mean(SpecialmetricOneProjUNet, axis = 1)
FPmetricAvgOneProjUNet = np.mean(FPmetricOneProjUNet, axis = 1)
F1ScoreAvgOneProjUNet = np.mean(F1ScoreOneProjUNet, axis = 1)
SegAccStdOneProjUNet = np.std(SegAccOneProjUNet, axis = 1)
SpecialmetricStdOneProjUNet = np.std(SpecialmetricOneProjUNet, axis = 1)
FPmetricStdOneProjUNet = np.std(FPmetricOneProjUNet, axis = 1)
F1ScoreStdOneProjUNet = np.std(F1ScoreOneProjUNet, axis = 1)

SegAccAvgMSD = np.mean(SegAccMSD, axis = 1)
SpecialmetricAvgMSD = np.mean(SpecialmetricMSD, axis = 1)
FPmetricAvgMSD = np.mean(FPmetricMSD, axis = 1)
F1ScoreAvgMSD = np.mean(F1ScoreMSD, axis = 1)
SegAccStdMSD = np.std(SegAccMSD, axis = 1)
SpecialmetricStdMSD = np.std(SpecialmetricMSD, axis = 1)
FPmetricStdMSD = np.std(FPmetricMSD, axis = 1)
F1ScoreStdMSD = np.std(F1ScoreMSD, axis = 1)

SegAccAvgOneProjMSD = np.mean(SegAccOneProjMSD, axis = 1)
SpecialmetricAvgOneProjMSD = np.mean(SpecialmetricOneProjMSD, axis = 1)
FPmetricAvgOneProjMSD = np.mean(FPmetricOneProjMSD, axis = 1)
F1ScoreAvgOneProjMSD = np.mean(F1ScoreOneProjMSD, axis = 1)
SegAccStdOneProjMSD = np.std(SegAccOneProjMSD, axis = 1)
SpecialmetricStdOneProjMSD = np.std(SpecialmetricOneProjMSD, axis = 1)
FPmetricStdOneProjMSD = np.std(FPmetricOneProjMSD, axis = 1)
F1ScoreStdOneProjMSD = np.std(F1ScoreOneProjMSD, axis = 1)


#Plot everything
plt.plot(NumObjs, SegAccAvgUNet, label='Workflow: Fixed 1800 radiographs (U-Net)', linestyle='solid', marker = '^', color = 'brown')
plt.plot(NumObjs, SegAccAvgMSD, label='Workflow: Fixed 1800 radiographs (MSD)', linestyle='solid', marker = 'o', color = 'red')
plt.plot(NumObjsOneProj, SegAccAvgOneProjUNet, label='Classical: 1 radiograph per object (U-Net)', linestyle='dotted', marker = '^', color = 'purple')
plt.plot(NumObjsOneProj, SegAccAvgOneProjMSD, label='Classical: 1 radiograph per object (MSD)', linestyle='dotted', marker = 'o', color = 'orange')
plt.legend(loc = 'lower right')
plt.xlabel('Number of training objects included')
plt.ylabel('Average class accuracy (%)')
plt.title('Average class accuracies for different number of training objects')
plt.ylim(bottom = 50, top = 100)
plt.xlim(left = 1, right = 60)
plt.fill_between(NumObjs, SegAccAvgUNet - SegAccStdUNet, SegAccAvgUNet + SegAccStdUNet, alpha = 0.2, color = 'brown')
plt.fill_between(NumObjs, SegAccAvgMSD - SegAccStdMSD, SegAccAvgMSD + SegAccStdMSD, alpha = 0.2, color = 'red')
plt.fill_between(NumObjsOneProj, SegAccAvgOneProjUNet - SegAccStdOneProjUNet, SegAccAvgOneProjUNet + SegAccStdOneProjUNet, alpha = 0.2, color = 'purple')
plt.fill_between(NumObjsOneProj, SegAccAvgOneProjMSD - SegAccStdOneProjMSD, SegAccAvgOneProjMSD + SegAccStdOneProjMSD, alpha = 0.2, color = 'orange')
plt.savefig(pathToPlots + 'Results_ManyvsOneProj_MSDUNET_' + str(Runs) + 'Avgs_AvgClassAcc_shaded.png', dpi=500)
plt.savefig(pathToPlots + 'Results_ManyvsOneProj_MSDUNET_' + str(Runs) + 'Avgs_AvgClassAcc_shaded.eps')
plt.show()



plt.plot(NumObjs, SpecialmetricAvgUNet, label='Workflow: Fixed 1800 radiographs (U-Net)', linestyle='solid', marker = '^', color = 'brown')
plt.plot(NumObjs, SpecialmetricAvgMSD, label='Workflow: Fixed 1800 radiographs (MSD)', linestyle='solid', marker = 'o', color = 'red')
plt.plot(NumObjsOneProj, SpecialmetricAvgOneProjUNet, label='Classical: 1 radiograph per object (U-Net)', linestyle='dotted', marker = '^', color = 'purple')
plt.plot(NumObjsOneProj, SpecialmetricAvgOneProjMSD, label='Classical: 1 radiograph per object (MSD)', linestyle='dotted', marker = 'o', color = 'orange')
plt.legend(loc = 'lower right')
plt.xlabel('Number of training objects included')
plt.ylabel('Object based detection rate (%)')
plt.title('Object based detection rate\nfor different number of training objects')
plt.ylim(bottom = 0, top = 100)
plt.xlim(left = 1, right = 60)
plt.fill_between(NumObjs, SpecialmetricAvgUNet - SpecialmetricStdUNet, SpecialmetricAvgUNet + SpecialmetricStdUNet, alpha = 0.2, color = 'brown')
plt.fill_between(NumObjs, SpecialmetricAvgMSD - SpecialmetricStdMSD, SpecialmetricAvgMSD + SpecialmetricStdMSD, alpha = 0.2, color = 'red')
plt.fill_between(NumObjsOneProj, SpecialmetricAvgOneProjUNet - SpecialmetricStdOneProjUNet, SpecialmetricAvgOneProjUNet + SpecialmetricStdOneProjUNet, alpha = 0.2, color = 'purple')
plt.fill_between(NumObjsOneProj, SpecialmetricAvgOneProjMSD - SpecialmetricStdOneProjMSD, SpecialmetricAvgOneProjMSD + SpecialmetricStdOneProjMSD, alpha = 0.2, color = 'orange')
plt.savefig(pathToPlots + 'Results_ManyvsOneProj_MSDUNET_' + str(Runs) + 'Avgs_DetAcc_shaded.png', dpi=500)
plt.savefig(pathToPlots + 'Results_ManyvsOneProj_MSDUNET_' + str(Runs) + 'Avgs_DetAcc_shaded.eps')
plt.show()


plt.plot(NumObjs, FPmetricAvgUNet, label='Workflow: Fixed 1800 radiographs (U-Net)', linestyle='solid', marker = '^', color = 'brown')
plt.plot(NumObjs, FPmetricAvgMSD, label='Workflow: Fixed 1800 radiographs (MSD)', linestyle='solid', marker = 'o', color = 'red')
plt.plot(NumObjsOneProj, FPmetricAvgOneProjUNet, label='Classical: 1 radiograph per object (U-Net)', linestyle='dotted', marker = '^', color = 'purple')
plt.plot(NumObjsOneProj, FPmetricAvgOneProjMSD, label='Classical: 1 radiograph per object (MSD)', linestyle='dotted', marker = 'o', color = 'orange')
plt.legend(loc = 'upper right')
plt.xlabel('Number of training objects included')
plt.ylabel('False positive detection rate (%)')
plt.title('Object based false positive detection rate\nfor different number of training objects')
plt.ylim(bottom = 0, top = 70)
plt.xlim(left = 1, right = 60)
plt.fill_between(NumObjs, FPmetricAvgUNet - FPmetricStdUNet, FPmetricAvgUNet + FPmetricStdUNet, alpha = 0.2, color = 'brown')
plt.fill_between(NumObjs, FPmetricAvgMSD - FPmetricStdMSD, FPmetricAvgMSD + FPmetricStdMSD, alpha = 0.2, color = 'red')
plt.fill_between(NumObjsOneProj, FPmetricAvgOneProjUNet - FPmetricStdOneProjUNet, FPmetricAvgOneProjUNet + FPmetricStdOneProjUNet, alpha = 0.2, color = 'purple')
plt.fill_between(NumObjsOneProj, FPmetricAvgOneProjMSD - FPmetricStdOneProjMSD, FPmetricAvgOneProjMSD + FPmetricStdOneProjMSD, alpha = 0.2, color = 'orange')
plt.savefig(pathToPlots + 'Results_ManyvsOneProj_MSDUNET_' + str(Runs) + 'Avgs_FPrate_shaded.png', dpi=500)
plt.savefig(pathToPlots + 'Results_ManyvsOneProj_MSDUNET_' + str(Runs) + 'Avgs_FPrate_shaded.eps')
plt.show()


plt.plot(NumObjs, F1ScoreAvgUNet, label='Workflow: Fixed 1800 radiographs (U-Net)', linestyle='solid', marker = '^', color = 'brown')
plt.plot(NumObjs, F1ScoreAvgMSD, label='Workflow: Fixed 1800 radiographs (MSD)', linestyle='solid', marker = 'o', color = 'red')
plt.plot(NumObjsOneProj, F1ScoreAvgOneProjUNet, label='Manual annotation: 1 radiograph per object (U-Net)', linestyle='dotted', marker = '^', color = 'purple')
plt.plot(NumObjsOneProj, F1ScoreAvgOneProjMSD, label='Manual annotation: 1 radiograph per object (MSD)', linestyle='dotted', marker = 'o', color = 'orange')
plt.legend(loc = 'lower right')
plt.xlabel('Number of training objects included')
plt.ylabel('F1 score (%)')
plt.title('F1 scores for different number of training objects')
plt.ylim(bottom = 0, top = 100)
plt.xlim(left = 1, right = 60)
plt.fill_between(NumObjs, F1ScoreAvgUNet - F1ScoreStdUNet, F1ScoreAvgUNet + F1ScoreStdUNet, alpha = 0.2, color = 'brown')
plt.fill_between(NumObjs, F1ScoreAvgMSD - F1ScoreStdMSD, F1ScoreAvgMSD + F1ScoreStdMSD, alpha = 0.2, color = 'red')
plt.fill_between(NumObjsOneProj, F1ScoreAvgOneProjUNet - F1ScoreStdOneProjUNet, F1ScoreAvgOneProjUNet + F1ScoreStdOneProjUNet, alpha = 0.2, color = 'purple')
plt.fill_between(NumObjsOneProj, F1ScoreAvgOneProjMSD - F1ScoreStdOneProjMSD, F1ScoreAvgOneProjMSD + F1ScoreStdOneProjMSD, alpha = 0.2, color = 'orange')
plt.savefig(pathToPlots + 'Results_ManyvsOneProj_MSDUNET_' + str(Runs) + 'Avgs_F1Score_shaded.png', dpi=500)
plt.savefig(pathToPlots + 'Results_ManyvsOneProj_MSDUNET_' + str(Runs) + 'Avgs_F1Score_shaded.eps')
plt.show()
