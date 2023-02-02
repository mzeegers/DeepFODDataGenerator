#This is code to support the assertions made in the paper (see section 4.8.): 
#for simulated experiments using 3D ground truth is similar to reconstructing and segmentating from its projections

#This code computes a wide variety of similarity measures (among these are the Jaccard similarity and MSE) between the 3D phantoms and the workflow-generated 3D segmentations,
# and between the 'perfect' projected segmentations and the workflow-generated projected segmentations
#The code assumes that PhantomGeneratorTrainandTest.py, PhantomProjectorTrainandTest.py, SpectralDataGenerator.py and ReconstructAndProject.py are carried out first
# and that the perfect projection segmentations are available in the /data/Numerical/GTProjectionsPerfectTrain/, the generated projection segmentations are available in the /data/Numerical/GTProjectionsTrain/ folders,
# the phantoms are available in the /data/Numerical/Objects3DTrain/ and that the reconstructions are availble in the /data/Numerical/Reconstructions/ folders

#NOTE: This code is only needed for subsequent comparison of the 'perfect' ground truth with the workflow-generated ground truth, and not for training with the neural networks (see paper section 4.8. for explanation)

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)


import numpy as np
import pyqtgraph as pq
import tifffile
from sklearn.metrics import jaccard_score, mean_squared_error


threshold3D = 0.04  #Segmentation threshold for the 3D reconstructions applied in the workflow

##Compute statistics for projected images: 'perfect' ground truth segmentations vs. workflow-generated ground truth segmentations

#Initialize 
AvgJaccardProj = 0
AvgJaccardProjNonPerfect = 0
AvgMeanSquareProj = 0
AvgMeanSquareProjNonPerfect = 0
count = 0
countProjNonPerfect = 0
lowestJaccardValProj = np.inf

#Paths to the images to compare
SegPath = '../../../data/Numerical/GTProjectionsPerfectTrain/'
GTPath = '../../../data/Numerical/GTProjectionsTrain/'

for inst in range(0,100):
    print("Object instance", inst)
    for ang in range(0,1800):
    
        #Load the files
        MatProjSeg = tifffile.imread(SegPath + 'Instance' + str(inst).zfill(3) + '/Instance' + str(inst).zfill(3) + 'Angle' + str(ang).zfill(4) + '.tiff')
        MatProjGT = tifffile.imread(GTPath + 'Instance' + str(inst).zfill(3) + '/Instance' + str(inst).zfill(3) + 'Angle' + str(ang).zfill(4) + '.tiff')
    
        JaccardScoreProj = jaccard_score(MatProjGT.flatten(), MatProjSeg.flatten())
        MeanSquareProj = mean_squared_error(MatProjGT.flatten(), MatProjSeg.flatten())
            
        if(JaccardScoreProj < lowestJaccardValProj):
            lowestJaccardObjIndexProj = inst
            lowestJaccardAngIndexProj = ang
            lowestJaccardValProj = JaccardScoreProj
            lowestJaccardSegProj = MatProjSeg
            lowestJaccardGTProj = MatProjGT
            
        AvgJaccardProj += JaccardScoreProj
        AvgMeanSquareProj += MeanSquareProj
        if(JaccardScoreProj != 1.0):
            countProjNonPerfect += 1
            AvgJaccardProjNonPerfect += JaccardScoreProj
            AvgMeanSquareProjNonPerfect += MeanSquareProj
        count += 1

print("Statistics for projections:")
AvgJaccardProj = AvgJaccardProj/count
AvgJaccardProjNonPerfect = AvgJaccardProjNonPerfect/countProjNonPerfect
AvgMeanSquareProj = AvgMeanSquareProj/count
AvgMeanSquareProjNonPerfect = AvgMeanSquareProjNonPerfect/countProjNonPerfect
print("Examples considered", count)
print("Number of perfect projections:", count - countProjNonPerfect)
print("Average Jaccard score remaining", AvgJaccardProjNonPerfect)
print("Average Jaccard score on projection", AvgJaccardProj)
print("Average mean square error remaining", AvgMeanSquareProjNonPerfect)
print("Average mean square error on projection", AvgMeanSquareProj)
        
print("Object with lowest projection Jaccard similarity:")
print("Object", lowestJaccardObjIndexProj,"at angle", lowestJaccardAngIndexProj, "with", lowestJaccardValProj)
pq.image(lowestJaccardSegProj, title = "Created projection")
pq.image(lowestJaccardGTProj, title = "GT")
input()


##Compute statistics for volumes: generated phantoms vs. workflow-generated 3D segmentations

AvgJaccardProj = 0
AvgJaccardProjNonPerfect = 0
AvgMeanSquareProj = 0
AvgMeanSquareProjNonPerfect = 0
count = 0
countProjNonPerfect = 0
lowestJaccardValProj = np.inf

SegPath = '../../../data/Numerical/Reconstructions/'
GTPath = '../../../data/Numerical/Objects3DTrain/'

for inst in range(0,100):
    print("Object instance", inst)
    MatProjSegVol = tifffile.imread(SegPath + str(inst).zfill(3) + 'reconstruction.tiff')
    MatProjGTVol = tifffile.imread(GTPath + str(inst).zfill(3) + 'instObj3D.tiff')
    
    MatProjSeg = np.zeros_like(MatProjSegVol)
    MatProjSeg[MatProjSegVol > threshold3D] = 1
    MatProjGT = np.zeros_like(MatProjGTVol)
    MatProjGT[MatProjGTVol == 2] = 1
    
    JaccardScoreProj = jaccard_score(MatProjGT.flatten(), MatProjSeg.flatten())
    MeanSquareProj = mean_squared_error(MatProjGT.flatten(), MatProjSeg.flatten())
            
    if(JaccardScoreProj < lowestJaccardValProj):
        lowestJaccardObjIndexProj = inst
        lowestJaccardValProj = JaccardScoreProj
        lowestJaccardSegProj = MatProjSeg
        lowestJaccardGTProj = MatProjGT
            
    AvgJaccardProj += JaccardScoreProj
    AvgMeanSquareProj += MeanSquareProj
    if(JaccardScoreProj != 1.0):
        countProjNonPerfect += 1
        AvgJaccardProjNonPerfect += JaccardScoreProj
        AvgMeanSquareProjNonPerfect += MeanSquareProj
    count += 1

print("Statistics for projections:")
AvgJaccardProj = AvgJaccardProj/count
AvgJaccardProjNonPerfect = AvgJaccardProjNonPerfect/countProjNonPerfect
AvgMeanSquareProj = AvgMeanSquareProj/count
AvgMeanSquareProjNonPerfect = AvgMeanSquareProjNonPerfect/countProjNonPerfect
print("Examples considered", count)
print("Number of perfect projections:", count - countProjNonPerfect)
print("Average Jaccard score remaining", AvgJaccardProjNonPerfect)
print("Average Jaccard score on projection", AvgJaccardProj)
print("Average mean square error remaining", AvgMeanSquareProjNonPerfect)
print("Average mean square error on projection", AvgMeanSquareProj)
        
print("Object with lowest projection Jaccard similarity:")
print("Object", lowestJaccardObjIndexProj, "with", lowestJaccardValProj)
pq.image(lowestJaccardSegProj, title = "Created projection")
pq.image(lowestJaccardGTProj, title = "GT")
input()
