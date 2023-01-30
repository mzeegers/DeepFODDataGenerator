#Generation of training data from CT scans

#This script carries out the workflow for reconstruction of CT data, segmentation and forward projections
#The results are the flat- and darkfield corrected data and their ground truth locations
#The code assumes that the CT data has been downloaded and stored in the '/data/CTData/' folder
#Link to download location for the CT data: https://zenodo.org/record/5866228
#The code can be modified according to any changes/enhancements in the data generation workflow
#The defaultValues function contains the parameters that can be changed
#Code is compatible with segmentations containing multiple labels
#Assumption: CT scanning angles are equidistantly distributed

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)


import astra
import cv2
import numpy as np
import os
import pylab
import pyqtgraph as pq      #pyqtgraph slices through the first axis of a 3D array
import scipy as sp
import scipy.ndimage
import tifffile
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)

#Folders containing the CT data and folders to write results to
DataPath = '../../../data/CTData/'
GTSavePath = '../../../data/TrainingDataExperimentalGT/'
DataSavePath = '../../../data/TrainingDataExperimental/'

InstanceBegin = 1
InstanceEnd = 2


class DataProcessor(object):
    def __init__(self):
        self.CorData = None
        self.rec = None
        self.Aseg = None
        self.defaultValues()


    def defaultValues(self):
        
        self.Angles = 1800                      #Number of angles in each CT scan
        self.detSizey = 760                     #Number of detector pixels in the CT scan in y direction
        self.detSizex = 956                     #Number of detector pixels in the CT scan in x direction
        self.threshold3D = 0.011                #Threshold paramter of thresholding segmentation 
        self.resizey = 128                      #Target image size in y direction
        self.resizex = 128                      #Target image size in x direction (note: aspect ratio not maintained)
        self.projThreshold = 0                  #Threshold in projections: all values large than this will be set to 1
        self.projThresholdAfterresize = 0.5     #Threshold in projections after resizing
        self.ALG = 'FDK_CUDA'                   #Reconstruction algorithm in astra
        self.ITER = 1                           #Number of iteration in reconstruction algorithm
        self.resizeDataFactor = 0.5             #Optional resizing of the data for memory purposes
        self.detPixelsy = int(0.5*self.detSizey)#Number of projection detector pixels after resizing the data in y direction
        self.detPixelsx = int(0.5*self.detSizex)#Number of projection detector pixels after resizing the data in x direction
        
        #Taken from CT settings file
        self.DETPIXSIZE = 0.149600              #CT scan: detector pixel size
        self.SOD = 441.396942                   #CT scan: source-object distance 
        self.SDD = 698.000977                   #CT scan: source-detector distance
        self.MAGN = self.SDD/self.SOD           #CT scan: magnification
        self.CONV = self.MAGN/self.DETPIXSIZE   #CT scan: conversion factor between length units and astra units


    #Collects all CT data files corresponding to an instance number, corrects it and saves the results
    def CollectandSaveData(self, Instance):
    
        #Point to the folder corresponding to the CT scan instance
        if(Instance <= 111):
            DataPathInstance = DataPath + 'Object' + str(Instance) + '_Scan20W/'
        else:
            DataPathInstance = DataPath + 'ManyObject' + str(Instance-111) + '_Scan20W/'
    
        #Load all data filenames
        filenames = sorted(os.listdir(DataPathInstance))

        #Select only filenames corresponding to the data
        substring = '.tif'
        files = [s for s in filenames if substring in s]

        #Find resolution
        resolution = tifffile.imread(DataPathInstance + files[0]).shape
        print("Detected image resolution:", resolution)

        #Darkfield data files
        darkfiles = [s for s in filenames if 'di' in s]
        #Flatfield dat files
        flatfiles = [s for s in filenames if 'io0' in s]
        #Main data files
        datafiles = [s for s in filenames if 'scan_0' in s]

        #Define array with zeros of (numberoffiles,res1,res2) and load the data and flatfields
        Data = np.zeros((len(datafiles), resolution[0], resolution[1]))

        #Load the darkfield images (and average)
        DarkData = np.zeros((resolution[0], resolution[1]))
        for f in darkfiles:
            print(f)
            DarkData += tifffile.imread(DataPathInstance + f)
        DarkData = DarkData/len(darkfiles)
            
        #Load the flatfield images (and average)
        FlatData = np.zeros((resolution[0], resolution[1]))
        for f in flatfiles:
            print(f)
            FlatData += tifffile.imread(DataPathInstance + f)
        FlatData = FlatData/len(flatfiles)

        #Load the remaining CT images and save the results after dark- and flatfield correction
        fileCounter = 0
            
        print("Loading and correcting...")
        for f in tqdm(datafiles):
            #Dark- and flatfield correction
            Data[fileCounter, :, :] = np.log((FlatData - DarkData)/(tifffile.imread(DataPathInstance + f) - DarkData))
            
            #Resize the data and save the corrected radiographs
            SaveData = cv2.resize(Data[fileCounter, :, :], dsize=(self.resizey, self.resizex), interpolation=cv2.INTER_CUBIC)
            tifffile.imsave(DataSavePath + '/Instance' + str(Instance).zfill(3) + '/Instance' + str(Instance).zfill(3) + 'Angle{:05d}Data.tiff'.format(fileCounter), SaveData.astype(np.float32))
                
            fileCounter += 1
        
        #Check for if the number of detected files is not correct
        if(fileCounter != self.Angles):
            print(fileCounter)
            print("Warning: No 1800 projections found! Input any key to continue")
            input()
            
        self.CorData = np.swapaxes(Data,0,1)

        #View the corrected data (uncomment if needed)
        #pq.image(self.CorData)
        #input()


    #Carry out a volume projection
    def forwardProjectSimple(self, ProjSeg):

        #Create volume geometry, angle distribution and projection geometry
        vol_geom = astra.create_vol_geom(ProjSeg.shape)   
        angles = np.linspace(0,360*np.pi/180, self.Angles, False)
        proj_geom = astra.create_proj_geom('cone', self.DETPIXSIZE*self.CONV, self.DETPIXSIZE*self.CONV, 760, 956, angles, self.SOD*self.CONV, (self.SDD-self.SOD)*self.CONV)

        #Carry out the projection
        proj_id, proj_data = astra.create_sino3d_gpu(ProjSeg, proj_geom, vol_geom)
        Proj = proj_data.swapaxes(0,1)

        #Clean up
        astra.projector.delete(proj_id)

        return Proj


    #Carry out projection and make it binary
    def createProjectionsSimple(self, ProjSeg, Binary):

        #Create the (non-thresholded) projections
        Res = self.forwardProjectSimple(ProjSeg)
            
        #Threshold the projected values
        if(Binary is True):
            Res[Res > self.projThreshold] = 1

        return Res


    #Create ground truth projection - collect labels in GroundTruth and include these in the forward projection
    def groundTruthProjectionSimple(self, GroundTruth, Binary):   
     
        #Select the labels in the segmentation to project
        print("Labels in segmentation to project:", GroundTruth)  
        ProjSeg = np.zeros_like(self.ASeg)
        for i in GroundTruth:
            ProjSeg[self.ASeg == i] = 1

        #Create the projections
        Proj = self.createProjectionsSimple(ProjSeg, Binary)

        return Proj


    # Reconstructs the object from the CT data
    def CTReconstruction(self):

        #Crop if needed in case of memory issues, uncomment if undesired
        print("Resizing data...")
        self.CorData = scipy.ndimage.zoom(self.CorData, (self.resizeDataFactor, 1, self.resizeDataFactor)) #middle argument represents angles
        print("Data size:", self.CorData.shape)

        #Create volume geometry, angle distribution and projection geometry
        vol_geom = astra.create_vol_geom(self.detPixelsx, self.detPixelsx, self.detPixelsx)
        angles = np.linspace(0,360*np.pi/180, self.Angles, False)
        proj_geom = astra.create_proj_geom('cone', self.DETPIXSIZE*self.CONV, self.DETPIXSIZE*self.CONV, self.detPixelsy, self.detPixelsx, angles, self.SOD*self.CONV, (self.SDD-self.SOD)*self.CONV)

        #Create the sinogram and choose the reconstruction algorithm
        sinogram_id = astra.data3d.create('-sino', proj_geom, self.CorData)

        ALG = 'FDK_CUDA'
        ITER = 1

        rec_id = astra.data3d.create('-vol', vol_geom)
        cfg = astra.astra_dict(self.ALG)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id

        alg_id = astra.algorithm.create(cfg)

        #Carry out the reconstruction
        print("Reconstructing...")
        astra.algorithm.run(alg_id, self.ITER)
        print("Finished")

        #Retrieve the reconstruction
        self.rec = astra.data3d.get(rec_id)

        #Cleaning up
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sinogram_id)

        #Show the reconstruction (uncomment if desired)
        #print(rec.shape)
        #pq.image(rec)
        #pq.image(np.swapaxes(rec,0,1))
        #pq.image(np.swapaxes(rec,0,2))
        #input()
    
    
    #Segmentation function
    def Segmentation(self):
    
        #Enlarge the reconstruction for proper segmentation
        ALarger = scipy.ndimage.zoom(self.rec, 2, order = 1)

        #Segmentation by simple thresholding
        self.ASeg = np.zeros_like(ALarger)
        self.ASeg[ALarger > self.threshold3D] = 1
    
    
    #Project the 3D segmentation back onto the detector and save the ground truth projections
    def ProjectSegmentation(self, Instance):
                    
        MatProj = self.groundTruthProjectionSimple([1], Binary = True)               

        print("Project under each angle...")
        for ang in tqdm(range(0,self.Angles)):
            
            #Resize the ground truth projections, make these binary by additional segmenation and save the corrected radiographs
            ASmaller = cv2.resize(MatProj[ang,:,:].astype(np.uint8), dsize=(self.resizey, self.resizex), interpolation=cv2.INTER_CUBIC)
            ASegProj = np.zeros_like(ASmaller)
            ASegProj[ASmaller >= self.projThresholdAfterresize] = 1
            ASegProj[ASmaller < self.projThresholdAfterresize] = 0              
            tifffile.imsave(GTSavePath + 'Instance' + str(Instance).zfill(3) + '/Instance' + str(Instance).zfill(3) + 'Angle' + str(ang).zfill(4) + '.tiff', ASegProj.astype(np.uint8))
    
    
    #Processes the entire dataset (data correction, CT reconstruction, segmentation, projection and processed data storage
    def ProcessData(self):

        #Loop over the CT scans
        for Instance in range(InstanceBegin, InstanceEnd):
            print("--- CT scan instance", Instance, " ---")
        
            #Create the necessary target folders
            os.makedirs(DataSavePath + '/Instance' + str(Instance).zfill(3) + '/', exist_ok=True)
            os.makedirs(GTSavePath + '/Instance' + str(Instance).zfill(3) + '/', exist_ok=True)
        
            #Run the necessary steps
            self.CollectandSaveData(Instance)            
            self.CTReconstruction()
            self.Segmentation()
            self.ProjectSegmentation(Instance)
            
            self.CorData = None
            self.rec = None
            self.Aseg = None


def main():
    Ph = DataProcessor()
    Ph.ProcessData()

if __name__ == "__main__":
    main()
