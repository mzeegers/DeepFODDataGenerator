#Plotting script for results from MSD and UNet networks tested on workflow generated data (with varying amounts of foreign objects)

# This script plots the results of MSD and UNet networks on the workflow-generated (experimental) testing data
# The plots indicate the difference in results when different number of foreign objects are contained in the CT scanned objects (few vs. many vs. mixed)
# The results are the plots files in the results/experimental/plots/ folder

# The code assumes that the csv files with MSD and UNet network results on training data are available in the results/experimental/quantitative/ folder

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
pathToPlots = '../../../results/experimental/plots/'

#Load relevant files
SegAccUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_AvgClassAcc.csv', delimiter=' ')
SpecialmetricUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_DetAcc.csv', delimiter=' ')
FPmetricUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_FPrate.csv', delimiter=' ')
F1ScoreUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_F1ScoreAcc.csv', delimiter=' ')
SegAccManyFOUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_ManyFO_AvgClassAcc.csv', delimiter=' ')
SpecialmetricManyFOUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_ManyFO_DetAcc.csv', delimiter=' ')
FPmetricManyFOUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_ManyFO_FPrate.csv', delimiter=' ')
F1ScoreManyFOUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_ManyFO_F1ScoreAcc.csv', delimiter=' ')
SegAccMixedUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_Mixed_AvgClassAcc.csv', delimiter=' ')
SpecialmetricMixedUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_Mixed_DetAcc.csv', delimiter=' ')
FPmetricMixedUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_Mixed_FPrate.csv', delimiter=' ')
F1ScoreMixedUNet = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_UNet_Mixed_F1ScoreAcc.csv', delimiter=' ')

SegAccMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_AvgClassAcc.csv', delimiter=' ')
SpecialmetricMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_DetAcc.csv', delimiter=' ')
FPmetricMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_FPrate.csv', delimiter=' ')
F1ScoreMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_F1ScoreAcc.csv', delimiter=' ')
SegAccManyFOMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_ManyFO_AvgClassAcc.csv', delimiter=' ')
SpecialmetricManyFOMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_ManyFO_DetAcc.csv', delimiter=' ')
FPmetricManyFOMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_ManyFO_FPrate.csv', delimiter=' ')
F1ScoreManyFOMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_ManyFO_F1ScoreAcc.csv', delimiter=' ')
SegAccMixedMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_Mixed_AvgClassAcc.csv', delimiter=' ')
SpecialmetricMixedMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_Mixed_DetAcc.csv', delimiter=' ')
FPmetricMixedMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_Mixed_FPrate.csv', delimiter=' ')
F1ScoreMixedMSD = np.loadtxt(curPath + pathToQuant + 'RawResults_Paper_MSD_Mixed_F1ScoreAcc.csv', delimiter=' ')

Runs = SegAccMSD.shape[1]

##Additional data
NumObjs = [1,2,3,4,5,7,10,20,30,40,50,60]
NumObjsManyFO = [1,2,3,4,5,7,10,20]
NumObjsMixed = [1,2,3,4,5,7,10,20,30,40]

#Compute averages and stds
SegAccAvgUNet = np.mean(SegAccUNet, axis = 1)
SpecialmetricAvgUNet = np.mean(SpecialmetricUNet, axis = 1)
FPmetricAvgUNet = np.mean(FPmetricUNet, axis = 1)
F1ScoreAvgUNet = np.mean(F1ScoreUNet, axis = 1)
SegAccStdUNet = np.std(SegAccUNet, axis = 1)
SpecialmetricStdUNet = np.std(SpecialmetricUNet, axis = 1)
FPmetricStdUNet = np.std(FPmetricUNet, axis = 1)
F1ScoreStdUNet = np.std(F1ScoreUNet, axis = 1)

SegAccAvgManyFOUNet = np.mean(SegAccManyFOUNet, axis = 1)
SpecialmetricAvgManyFOUNet = np.mean(SpecialmetricManyFOUNet, axis = 1)
FPmetricAvgManyFOUNet = np.mean(FPmetricManyFOUNet, axis = 1)
F1ScoreAvgManyFOUNet = np.mean(F1ScoreManyFOUNet, axis = 1)
SegAccStdManyFOUNet = np.std(SegAccManyFOUNet, axis = 1)
SpecialmetricStdManyFOUNet = np.std(SpecialmetricManyFOUNet, axis = 1)
FPmetricStdManyFOUNet = np.std(FPmetricManyFOUNet, axis = 1)
F1ScoreStdManyFOUNet = np.std(F1ScoreManyFOUNet, axis = 1)

SegAccAvgMixedUNet = np.mean(SegAccMixedUNet, axis = 1)
SpecialmetricAvgMixedUNet = np.mean(SpecialmetricMixedUNet, axis = 1)
FPmetricAvgMixedUNet = np.mean(FPmetricMixedUNet, axis = 1)
F1ScoreAvgMixedUNet = np.mean(F1ScoreMixedUNet, axis = 1)
SegAccStdMixedUNet = np.std(SegAccMixedUNet, axis = 1)
SpecialmetricStdMixedUNet = np.std(SpecialmetricMixedUNet, axis = 1)
FPmetricStdMixedUNet = np.std(FPmetricMixedUNet, axis = 1)
F1ScoreStdMixedUNet = np.std(F1ScoreMixedUNet, axis = 1)

SegAccAvgMSD = np.mean(SegAccMSD, axis = 1)
SpecialmetricAvgMSD = np.mean(SpecialmetricMSD, axis = 1)
FPmetricAvgMSD = np.mean(FPmetricMSD, axis = 1)
F1ScoreAvgMSD = np.mean(F1ScoreMSD, axis = 1)
SegAccStdMSD = np.std(SegAccMSD, axis = 1)
SpecialmetricStdMSD = np.std(SpecialmetricMSD, axis = 1)
FPmetricStdMSD = np.std(FPmetricMSD, axis = 1)
F1ScoreStdMSD = np.std(F1ScoreMSD, axis = 1)

SegAccAvgManyFOMSD = np.mean(SegAccManyFOMSD, axis = 1)
SpecialmetricAvgManyFOMSD = np.mean(SpecialmetricManyFOMSD, axis = 1)
FPmetricAvgManyFOMSD = np.mean(FPmetricManyFOMSD, axis = 1)
F1ScoreAvgManyFOMSD = np.mean(F1ScoreManyFOMSD, axis = 1)
SegAccStdManyFOMSD = np.std(SegAccManyFOMSD, axis = 1)
SpecialmetricStdManyFOMSD = np.std(SpecialmetricManyFOMSD, axis = 1)
FPmetricStdManyFOMSD = np.std(FPmetricManyFOMSD, axis = 1)
F1ScoreStdManyFOMSD = np.std(F1ScoreManyFOMSD, axis = 1)

SegAccAvgMixedMSD = np.mean(SegAccMixedMSD, axis = 1)
SpecialmetricAvgMixedMSD = np.mean(SpecialmetricMixedMSD, axis = 1)
FPmetricAvgMixedMSD = np.mean(FPmetricMixedMSD, axis = 1)
F1ScoreAvgMixedMSD = np.mean(F1ScoreMixedMSD, axis = 1)
SegAccStdMixedMSD = np.std(SegAccMixedMSD, axis = 1)
SpecialmetricStdMixedMSD = np.std(SpecialmetricMixedMSD, axis = 1)
FPmetricStdMixedMSD = np.std(FPmetricMixedMSD, axis = 1)
F1ScoreStdMixedMSD = np.std(F1ScoreMixedMSD, axis = 1)


#Plot everything
plt.plot(NumObjs, SegAccAvgUNet, label='Few foreign objects (U-Net)', linestyle='solid', marker = '^', color = 'brown')
plt.plot(NumObjs, SegAccAvgMSD, label='Few foreign objects (MSD)', linestyle='solid', marker = 'o', color = 'red')
plt.plot(NumObjsManyFO, SegAccAvgManyFOUNet, label='Many foreign objects (U-Net)', linestyle='dotted', marker = '^', color = 'grey')
plt.plot(NumObjsManyFO, SegAccAvgManyFOMSD, label='Many foreign objects (MSD)', linestyle='dotted', marker = 'o', color = 'blue')
plt.plot(NumObjsMixed, SegAccAvgMixedUNet, label='Mixed (U-Net)', linestyle='dashed', marker = '^', color = 'turquoise')
plt.plot(NumObjsMixed, SegAccAvgMixedMSD, label='Mixed (MSD)', linestyle='dashed', marker = 'o', color = 'green')
plt.legend(loc = 'lower right')
plt.xlabel('Number of training objects included')
plt.ylabel('Average class accuracy (%)')
plt.title('Average class accuracies for different number of training objects')
plt.ylim(bottom = 50, top = 100)
plt.xlim(left = 1, right = 60)
plt.fill_between(NumObjs, SegAccAvgUNet - SegAccStdUNet, SegAccAvgUNet + SegAccStdUNet, alpha = 0.2, color = 'brown')
plt.fill_between(NumObjs, SegAccAvgMSD - SegAccStdMSD, SegAccAvgMSD + SegAccStdMSD, alpha = 0.2, color = 'red')
plt.fill_between(NumObjsManyFO, SegAccAvgManyFOUNet - SegAccStdManyFOUNet, SegAccAvgManyFOUNet + SegAccStdManyFOUNet, alpha = 0.2, color = 'grey')
plt.fill_between(NumObjsManyFO, SegAccAvgManyFOMSD - SegAccStdManyFOMSD, SegAccAvgManyFOMSD + SegAccStdManyFOMSD, alpha = 0.2, color = 'blue')
plt.fill_between(NumObjsMixed, SegAccAvgMixedUNet - SegAccStdMixedUNet, SegAccAvgMixedUNet + SegAccStdMixedUNet, alpha = 0.2, color = 'turquoise')
plt.fill_between(NumObjsMixed, SegAccAvgMixedMSD - SegAccStdMixedMSD, SegAccAvgMixedMSD + SegAccStdMixedMSD, alpha = 0.2, color = 'green')
plt.savefig(pathToPlots + '/Results_DiffFOAmounts_MSDUNET_' + str(Runs) + 'Avgs_AvgClassAcc_shaded.png', dpi=500)
plt.savefig(pathToPlots + '/Results_DiffFOAmounts_MSDUNET_' + str(Runs) + 'Avgs_AvgClassAcc_shaded.eps')
plt.show()



plt.plot(NumObjs, SpecialmetricAvgUNet, label='Few foreign objects (U-Net)', linestyle='solid', marker = '^', color = 'brown')
plt.plot(NumObjs, SpecialmetricAvgMSD, label='Few foreign objects (MSD)', linestyle='solid', marker = 'o', color = 'red')
plt.plot(NumObjsManyFO, SpecialmetricAvgManyFOUNet, label='Many foreign objects (U-Net)', linestyle='dotted', marker = '^', color = 'grey')
plt.plot(NumObjsManyFO, SpecialmetricAvgManyFOMSD, label='Many foreign objects (MSD)', linestyle='dotted', marker = 'o', color = 'blue')
plt.plot(NumObjsMixed, SpecialmetricAvgMixedUNet, label='Mixed (U-Net)', linestyle='dashed', marker = '^', color = 'turquoise')
plt.plot(NumObjsMixed, SpecialmetricAvgMixedMSD, label='Mixed (MSD)', linestyle='dashed', marker = 'o', color = 'green')
plt.legend(loc = 'lower right')
plt.xlabel('Number of training objects included')
plt.ylabel('Object based detection rate (%)')
plt.title('Object based detection rate\nfor different number of training objects')
plt.ylim(bottom = 0, top = 100)
plt.xlim(left = 1, right = 60)
plt.fill_between(NumObjs, SpecialmetricAvgUNet - SpecialmetricStdUNet, SpecialmetricAvgUNet + SpecialmetricStdUNet, alpha = 0.2, color = 'brown')
plt.fill_between(NumObjs, SpecialmetricAvgMSD - SpecialmetricStdMSD, SpecialmetricAvgMSD + SpecialmetricStdMSD, alpha = 0.2, color = 'red')
plt.fill_between(NumObjsManyFO, SpecialmetricAvgManyFOUNet - SpecialmetricStdManyFOUNet, SpecialmetricAvgManyFOUNet + SpecialmetricStdManyFOUNet, alpha = 0.2, color = 'grey')
plt.fill_between(NumObjsManyFO, SpecialmetricAvgManyFOMSD - SpecialmetricStdManyFOMSD, SpecialmetricAvgManyFOMSD + SpecialmetricStdManyFOMSD, alpha = 0.2, color = 'blue')
plt.fill_between(NumObjsMixed, SpecialmetricAvgMixedUNet - SpecialmetricStdMixedUNet, SpecialmetricAvgMixedUNet + SpecialmetricStdMixedUNet, alpha = 0.2, color = 'turquoise')
plt.fill_between(NumObjsMixed, SpecialmetricAvgMixedMSD - SpecialmetricStdMixedMSD, SpecialmetricAvgMixedMSD + SpecialmetricStdMixedMSD, alpha = 0.2, color = 'green')
plt.savefig(pathToPlots + 'Results_DiffFOAmounts_MSDUNET_' + str(Runs) + 'Avgs_DetAcc_shaded.png', dpi=500)
plt.savefig(pathToPlots + 'Results_DiffFOAmounts_MSDUNET_' + str(Runs) + 'Avgs_DetAcc_shaded.eps')
plt.show()


plt.plot(NumObjs, FPmetricAvgUNet, label='Few foreign objects (U-Net)', linestyle='solid', marker = '^', color = 'brown')
plt.plot(NumObjs, FPmetricAvgMSD, label='Few foreign objects (MSD)', linestyle='solid', marker = 'o', color = 'red')
plt.plot(NumObjsManyFO, FPmetricAvgManyFOUNet, label='Many foreign objects (U-Net)', linestyle='dotted', marker = '^', color = 'grey')
plt.plot(NumObjsManyFO, FPmetricAvgManyFOMSD, label='Many foreign objects (MSD)', linestyle='dotted', marker = 'o', color = 'blue')
plt.plot(NumObjsMixed, FPmetricAvgMixedUNet, label='Mixed (U-Net)', linestyle='dashed', marker = '^', color = 'turquoise')
plt.plot(NumObjsMixed, FPmetricAvgMixedMSD, label='Mixed (MSD)', linestyle='dashed', marker = 'o', color = 'green')
plt.legend(loc = 'upper right')
plt.xlabel('Number of training objects included')
plt.ylabel('False positive detection rate (%)')
plt.title('Object based false positive detection rate\nfor different number of training objects')
plt.ylim(bottom = 0, top = 70)
plt.xlim(left = 1, right = 60)
plt.fill_between(NumObjs, FPmetricAvgUNet - FPmetricStdUNet, FPmetricAvgUNet + FPmetricStdUNet, alpha = 0.2, color = 'brown')
plt.fill_between(NumObjs, FPmetricAvgMSD - FPmetricStdMSD, FPmetricAvgMSD + FPmetricStdMSD, alpha = 0.2, color = 'red')
plt.fill_between(NumObjsManyFO, FPmetricAvgManyFOUNet - FPmetricStdManyFOUNet, FPmetricAvgManyFOUNet + FPmetricStdManyFOUNet, alpha = 0.2, color = 'grey')
plt.fill_between(NumObjsManyFO, FPmetricAvgManyFOMSD - FPmetricStdManyFOMSD, FPmetricAvgManyFOMSD + FPmetricStdManyFOMSD, alpha = 0.2, color = 'blue')
plt.fill_between(NumObjsMixed, FPmetricAvgMixedUNet - FPmetricStdMixedUNet, FPmetricAvgMixedUNet + FPmetricStdMixedUNet, alpha = 0.2, color = 'turquoise')
plt.fill_between(NumObjsMixed, FPmetricAvgMixedMSD - FPmetricStdMixedMSD, FPmetricAvgMixedMSD + FPmetricStdMixedMSD, alpha = 0.2, color = 'green')
plt.savefig(pathToPlots + 'Results_DiffFOAmounts_MSDUNET_' + str(Runs) + 'Avgs_FPrate_COMBINED_NEW_shaded.png', dpi=500)
plt.savefig(pathToPlots + 'Results_DiffFOAmounts_MSDUNET_' + str(Runs) + 'Avgs_FPrate_COMBINED_NEW_shaded.eps')
plt.show()

plt.plot(NumObjs, F1ScoreAvgUNet, label='Few foreign objects (U-Net)', linestyle='solid', marker = '^', color = 'brown')
plt.plot(NumObjs, F1ScoreAvgMSD, label='Few foreign objects (MSD)', linestyle='solid', marker = 'o', color = 'red')
plt.plot(NumObjsManyFO, F1ScoreAvgManyFOUNet, label='Many foreign objects (U-Net)', linestyle='dotted', marker = '^', color = 'grey')
plt.plot(NumObjsManyFO, F1ScoreAvgManyFOMSD, label='Many foreign objects (MSD)', linestyle='dotted', marker = 'o', color = 'blue')
plt.plot(NumObjsMixed, F1ScoreAvgMixedUNet, label='Mixed (U-Net)', linestyle='dashed', marker = '^', color = 'turquoise')
plt.plot(NumObjsMixed, F1ScoreAvgMixedMSD, label='Mixed (MSD)', linestyle='dashed', marker = 'o', color = 'green')
plt.legend(loc = 'lower right')
plt.xlabel('Number of training objects included')
plt.ylabel('F1 score (%)')
plt.title('F1 scores for different number of training objects')
plt.ylim(bottom = 0, top = 100)
plt.xlim(left = 1, right = 60)
plt.fill_between(NumObjs, F1ScoreAvgUNet - F1ScoreStdUNet, F1ScoreAvgUNet + F1ScoreStdUNet, alpha = 0.2, color = 'brown')
plt.fill_between(NumObjs, F1ScoreAvgMSD - F1ScoreStdMSD, F1ScoreAvgMSD + F1ScoreStdMSD, alpha = 0.2, color = 'red')
plt.fill_between(NumObjsManyFO, F1ScoreAvgManyFOUNet - F1ScoreStdManyFOUNet, F1ScoreAvgManyFOUNet + F1ScoreStdManyFOUNet, alpha = 0.2, color = 'grey')
plt.fill_between(NumObjsManyFO, F1ScoreAvgManyFOMSD - F1ScoreStdManyFOMSD, F1ScoreAvgManyFOMSD + F1ScoreStdManyFOMSD, alpha = 0.2, color = 'blue')
plt.fill_between(NumObjsMixed, F1ScoreAvgMixedUNet - F1ScoreStdMixedUNet, F1ScoreAvgMixedUNet + F1ScoreStdMixedUNet, alpha = 0.2, color = 'turquoise')
plt.fill_between(NumObjsMixed, F1ScoreAvgMixedMSD - F1ScoreStdMixedMSD, F1ScoreAvgMixedMSD + F1ScoreStdMixedMSD, alpha = 0.2, color = 'green')
plt.savefig(pathToPlots + 'Results_DiffFOAmounts_MSDUNET_' + str(Runs) + 'Avgs_F1Score_COMBINED_shaded.png', dpi=500)
plt.savefig(pathToPlots + 'Results_DiffFOAmounts_MSDUNET_' + str(Runs) + 'Avgs_F1Score_COMBINED_shaded.eps')
plt.show()
