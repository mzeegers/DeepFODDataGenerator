#Projection of 3D phantom materials (including perfect ground truth creation) for demonstration of the workflow

#This script carries out the material projections from 3D phantoms, and generated the 'perfect' ground truth (i.e. directly projected from the phantoms) 
#The results are the material projections and correpsponding ground truth projections for the training set
# as well as material projections (2 perpendicular ones per object) and corresponding ground truth projections for the test set
#The code assumes that the PhantomGeneratorTrainandTest.py script has been carried out to generate the phantoms.

#The defaultValues function contains the parameters that can be changed
#Code is compatible with segmentations containing multiple labels
#Assumption: CT scanning angles are equidistantly distributed

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)


import astra
import numpy as np
import operator
import os
import pyqtgraph as pq      #pyqtgraph slices through the first axis of a 3D array
import random
import tifffile

np.set_printoptions(threshold=np.inf)

path = '../../../data/Numerical/'

InstanceBegin = 0
InstanceEnd = 500

class Projector(object):
    def __init__(self):
        self.Ph = None
        self.defaultValues()


    def defaultValues(self):
    
        self.Angles = 1800      #Number of (equidistant) angles to produce
        self.detSizey = 128     #Target image size in y direction
        self.detSizex = 128     #Target image size in y direction
        self.projThreshold = 0  #Threshold in projections: all values large than this will be set to 1
        self.DETSIZE = 1
        self.SOD = 10000
        self.SDD = 11000
        self.MAGN = self.SDD/self.SOD
        self.CONV = self.MAGN/self.DETSIZE


    #Carry out a material volume projection
    def forwardProjectSimple(self, ProjSeg):

        #Create volume geometry, angle distribution and projection geometry
        vol_geom = astra.create_vol_geom(ProjSeg.shape)   
        angles = np.linspace(0,360*np.pi/180, self.Angles, False)
        proj_geom = astra.create_proj_geom('cone', self.DETSIZE*self.CONV, self.DETSIZE*self.CONV, self.detSizey, self.detSizex, angles, self.SOD*self.CONV, (self.SDD-self.SOD)*self.CONV)

        #Carry out the projection
        proj_id, proj_data = astra.create_sino3d_gpu(ProjSeg, proj_geom, vol_geom)
        Proj = proj_data.swapaxes(0,1)

        #Clean up
        astra.data3d.delete(proj_id)

        return Proj


    #Carry out projection and make it binary
    def showViewSimple(self, ProjSeg, Binary):

        #Create the (non-thresholded) projections
        Res = self.forwardProjectSimple(ProjSeg)
        
        #Threshold the projected values
        if(Binary is True):
            Res[Res > self.projThreshold] = 1

        return Res

    #Create material projections - collect labels in GroundTruth and include these in the forward projection
    def groundTruthProjectionSimple(self, GroundTruth, Binary):    
    
        #Select the labels in the segmentation to project
        print("Labels in segmentation to project:", GroundTruth)  
        ProjSeg = np.zeros_like(self.Ph)
        for i in GroundTruth:
            ProjSeg[self.Ph == i] = 1

        #Create the projections
        Proj = self.showViewSimple(ProjSeg, Binary)

        return Proj
        
        
    #Project all materials from all input objects and save these for each material together
    def ProjectMaterials(self):
       
        os.makedirs(path + '/MaterialProjectionsTrain/', exist_ok=True)
        os.makedirs(path + '/GTProjectionsPerfectTrain/', exist_ok=True)
        os.makedirs(path + '/MaterialProjectionsTest/', exist_ok=True)
        os.makedirs(path + '/GTProjectionsPerfectTest/', exist_ok=True)
                
        for Instance in range(InstanceBegin, InstanceEnd):
            print("--- Object instance", Instance, " ---")

            if (Instance < 100):
                os.makedirs(path + '/MaterialProjectionsTrain/Instance' + str(Instance).zfill(3) + '/', exist_ok=True)
                os.makedirs(path + '/GTProjectionsPerfectTrain/Instance' + str(Instance).zfill(3) + '/', exist_ok=True)
                
                self.Ph = tifffile.imread(path + 'Objects3DTrain/' + str(Instance).zfill(3) + 'instObj3D.tiff')

                MatProj = np.zeros((5, self.Angles, self.detSizey, self.detSizex), dtype = np.float32)
                for i in range(1,4):
                    MatProj[i-1,:,:,:] = self.groundTruthProjectionSimple([i], Binary = False)               

                GTProj = self.groundTruthProjectionSimple([2,3], Binary = True)
                print("Saving material projections and ground truth...")
                for ang in range(0,self.Angles):
                    tifffile.imsave(path + '/MaterialProjectionsTrain/Instance' + str(Instance).zfill(3) + '/Instance' + str(Instance).zfill(3) + 'Angle' + str(ang).zfill(5) + '.tiff', MatProj[:,ang,:,:].astype(np.float32))
                    tifffile.imsave(path + '/GTProjectionsPerfectTrain/Instance' + str(Instance).zfill(3) + '/Instance' + str(Instance).zfill(3) + 'Angle' + str(ang).zfill(4) + '.tiff', GTProj[ang,:,:].astype(np.uint8))
                    
            else:
                os.makedirs(path + '/MaterialProjectionsTest/Instance' + str(Instance-100).zfill(3) + '/', exist_ok=True)
                os.makedirs(path + '/GTProjectionsPerfectTest/Instance' + str(Instance-100).zfill(3) + '/', exist_ok=True)
  
                np.random.seed(Instance-100)
                random.seed(Instance-100)
                
                self.Ph = tifffile.imread(path + 'Objects3DTest/' + str(Instance-100).zfill(3) + 'instObj3D.tiff')
                
                MatProj = np.zeros((5, self.Angles, self.detSizey, self.detSizex), dtype = np.float32)
                for i in range(1,4):
                    MatProj[i-1,:,:,:] = self.groundTruthProjectionSimple([i], Binary = False)               
                
                GTProj = self.groundTruthProjectionSimple([2,3], Binary = True)
                
                #Pick random angle
                ang = np.random.randint(0, self.Angles)
                #Select an orthogonal angle
                ang2 = (ang + 450)%self.Angles
                
                print("Saving material projections and ground truth...")
                tifffile.imsave(path + '/MaterialProjectionsTest/Instance' + str(Instance-100).zfill(3) + '/Instance' + str(Instance-100).zfill(3) + 'Angle' + str(ang).zfill(5) + '.tiff', MatProj[:,ang,:,:].astype(np.float32))
                tifffile.imsave(path + '/MaterialProjectionsTest/Instance' + str(Instance-100).zfill(3) + '/Instance' + str(Instance-100).zfill(3) + 'Angle' + str(ang2).zfill(5) + '.tiff', MatProj[:,ang2,:,:].astype(np.float32))
                
                tifffile.imsave(path + '/GTProjectionsPerfectTest/Instance' + str(Instance-100).zfill(3) + '/Instance' + str(Instance-100).zfill(3) + 'Angle' + str(ang).zfill(4) + '.tiff', GTProj[ang,:,:].astype(np.uint8))
                tifffile.imsave(path + '/GTProjectionsPerfectTest/Instance' + str(Instance-100).zfill(3) + '/Instance' + str(Instance-100).zfill(3) + 'Angle' + str(ang2).zfill(4) + '.tiff', GTProj[ang2,:,:].astype(np.uint8))

def main():
    Ph = Projector()
    Ph.ProjectMaterials()

if __name__ == "__main__":
    main()
