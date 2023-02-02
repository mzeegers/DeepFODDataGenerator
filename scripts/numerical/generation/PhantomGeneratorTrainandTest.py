#Generation of 3D phantoms for demonstration of the workflow

#This script carries out the generation of 3D objects containing foreign objects with to demonstrate the workflow on
#The results are the 3D objects (rotated cubes with cut-off edges and one or two small foreign objects) saved in the data/NumbericalObjects3D folder

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)


import numpy as np
import os
import pickle
import random
import scipy.ndimage
import tifffile

#Set random seeds
random.seed(0)
np.random.seed(seed = 0)

n = 128 #Size of the entire phantom space. Note: many other values are hard-coded te keep the code readable
Instances = 500 #Number of generated objects

#Set output path of the generated 3D volumes
outputPath = '../../../data/Numerical/' #/export/scratch3/zeegers/AutomatedFODProjectSmallManyTouchNew/'
os.makedirs(outputPath + 'Objects3DTrain/', exist_ok=True)
if(Instances >= 100):
    os.makedirs(outputPath + 'Objects3DTrain/', exist_ok=True)

#Keep track of generated objects with more than 1 (i.e. 2) foreign objects
DoublesList = []

#Create a seperation in 3D space by means of a plane defined by three points (p1, p2 and p3)
def planeFromPoints(p1, p2, p3, xx, yy, zz, zr, yr, xr):
        #Compute two vectors in the plane
        v1 = p3 - p1
        v2 = p2 - p1
        #Compute the croos product (a vector normal to the plane)
        cp = np.cross(v1, v2)
        a, b, c = cp
        #Compute d = a * x + b * y + c * z with p3 which equals d
        d = np.dot(cp, p3)
        #Determine and return the points that lie on the different sides of the plane
        cut = np.zeros((n,n,n), dtype=np.uint8)
        if(zr*yr*xr > 0):
            cut = a*xx+b*yy+c*zz > d
        else:
            cut = a*xx+b*yy+c*zz < d
        return cut

for inst in range(0, Instances):

    print("Instance", inst)

    #Create the main cube of the object
    cube = np.zeros((n,n,n), dtype=np.uint8)
    cube[32:96, 32:96, 32:96] = 1

    #Create planes that randomly take off the edges of the cube
    xx,yy,zz = np.mgrid[-64:64,-64:64,-64:64]

    hsz = int(64/2)
    qsz = int(64/4)
    esz = int(64/8)
    repeat = 1          #Number of repetitions to take off edges 

    #Create 8 planes that take off the corners
    #Loop over all corner points
    for zr in [1,-1]:
        for yr in [1,-1]:
            for xr in [1,-1]: 
                for rep in (0, repeat+1):

                    #Take random locations on the outgoing edges on the corner point
                    Ed1z = np.random.randint(0,qsz)
                    Ed1y = np.random.randint(0,qsz)
                    Ed1x = np.random.randint(0,qsz)

                    #Compute the corresponding points on the edges
                    p1 = np.array([Ed1z, yr*hsz, xr*hsz])
                    p2 = np.array([zr*hsz, yr*Ed1y, xr*hsz])
                    p3 = np.array([zr*hsz, yr*hsz, xr*Ed1x])

                    #Get the plane and the corresponding division of the space
                    cut = planeFromPoints(p1,p2,p3,xx,yy,zz,zr,yr,xr)

                    #Compute the leftover cube
                    cube = np.logical_and(cut,cube)

    #Create another 8*3=24 planes that take off the corners as well but under a larger angle
    for zr in [1,-1]:
        for yr in [1,-1]:
            for xr in [1,-1]: 
         
                #z-direction
                Ed1y = np.random.randint(hsz-qsz,hsz)
                Ed1x = np.random.randint(hsz-qsz,hsz)

                p1 = np.array([-1*zr*hsz, yr*Ed1y, xr*hsz])
                p2 = np.array([zr*hsz, yr*Ed1y, xr*hsz])
                p3 = np.array([-1*zr*hsz, yr*hsz, xr*Ed1x])

                cut = planeFromPoints(p1,p2,p3,xx,yy,zz,zr,yr,xr)
                cube = np.logical_and(cut,cube)

                #y-direction
                Ed1z = np.random.randint(hsz-qsz,hsz)
                Ed1x = np.random.randint(hsz-qsz,hsz)

                p1 = np.array([zr*hsz , -1*yr*hsz, xr*Ed1x])
                p2 = np.array([zr*hsz, yr*hsz, xr*Ed1x])
                p3 = np.array([zr*Ed1z, -1*yr*hsz, xr*hsz])

                cut = planeFromPoints(p1,p2,p3,xx,yy,zz,zr,yr,xr)
                cube = np.logical_and(cut,cube)
                
                #x-direction
                Ed1z = np.random.randint(hsz-qsz,hsz)
                Ed1y = np.random.randint(hsz-qsz,hsz)

                p1 = np.array([zr*Ed1z, yr*hsz, -1*xr*hsz])
                p2 = np.array([zr*Ed1z, yr*hsz, xr*hsz])
                p3 = np.array([zr*hsz, yr*Ed1y, -1*xr*hsz])

                cut = planeFromPoints(p1,p2,p3,xx,yy,zz,zr,yr,xr)
                cube = np.logical_and(cut,cube)

    #Rotate the background object with random angles
    cube = cube.astype(np.float32)
    ang1 = np.random.randint(0,90)
    ang2 = np.random.randint(0,90)
    ang3 = np.random.randint(0,90)
    cube = scipy.ndimage.rotate(cube, ang1, axes=(1,2), reshape = False, order=5)
    cube = scipy.ndimage.rotate(cube, ang2, axes=(0,2), reshape = False, order=5)
    cube = scipy.ndimage.rotate(cube, ang3, axes=(0,1), reshape = False, order=5)
    #Threshold the noninteger values resulting from rotation
    cube[cube >= 0.5] = 1
    cube[cube < 0.5] = 0

    ###Now make a ellipsoid (first foreign objects) with random size
    ell = np.zeros((12,12,12), dtype=np.float32)
    a = np.random.randint(3,7)
    b = np.random.randint(3,7)
    c = np.random.randint(3,7)
    xxl,yyl,zzl = np.mgrid[-8:8,-8:8,-8:8]
    ell = (xxl/a)**2+(yyl/b)**2+(zzl/c)**2<1
    ell = ell.astype(np.float32)

    #Rotate the foreign objecct with random angles
    ang1 = np.random.randint(0,90)
    ang2 = np.random.randint(0,90)
    ang3 = np.random.randint(0,90)
    ell = scipy.ndimage.rotate(ell, ang1, axes=(1,2), reshape = False)
    ell = scipy.ndimage.rotate(ell, ang2, axes=(0,2), reshape = False)
    ell = scipy.ndimage.rotate(ell, ang3, axes=(0,1), reshape = False)
    #Remove the noninteger values resulting from rotation
    ell[ell >= 0.5] = 1
    ell[ell < 0.5] = 0

    ###Now place this object somewhere in the object
    Object = np.copy(cube)
    #Take random locations
    pz = np.random.randint(24,102)
    py = np.random.randint(24,102)
    px = np.random.randint(24,102)
    #Check if foreign object is (partially) included in the main object
    while(not np.any(np.array([Object[pz-8:pz+8, py-8:py+8, px-8:px+8] == 1]) & np.array([ell == 1]))):
        pz = np.random.randint(24,102)
        py = np.random.randint(24,102)
        px = np.random.randint(24,102)

    #Insert first foreign object in object with label 2
    FO = np.copy(ell)
    FO[FO > 0] = 2
    Object = Object.astype(np.float32)
    Object[pz-8:pz+8, py-8:py+8, px-8:px+8][FO>0] = FO[FO > 0]


    ###Now create a second ellipsoid (second foreign objects) with random size
    ell2 = np.zeros((12,12,12), dtype=np.float32)
    a = np.random.randint(3,7)
    b = np.random.randint(3,7)
    c = np.random.randint(3,7)
    xxl,yyl,zzl = np.mgrid[-8:8,-8:8,-8:8]
    ell2 = (xxl/a)**2+(yyl/b)**2+(zzl/c)**2<1
    ell2 = ell.astype(np.float32) #Note: ell instead of ell2 is an error but retained to presicely replicate the paper results

    #Rotate the foreign objecct with random angles
    ang1 = np.random.randint(0,90)
    ang2 = np.random.randint(0,90)
    ang3 = np.random.randint(0,90)
    ell2 = scipy.ndimage.rotate(ell2, ang1, axes=(1,2), reshape = False)
    ell2 = scipy.ndimage.rotate(ell2, ang2, axes=(0,2), reshape = False)
    ell2 = scipy.ndimage.rotate(ell2, ang2, axes=(0,1), reshape = False)    
    #Remove the noninteger values resulting from rotation
    ell2[ell2 >= 0.5] = 1
    ell2[ell2 < 0.5] = 0
    
    ###Place this second foreign object somewhere in the object (not overlapping the first foreign object):
    if(random.uniform(0, 1) > 0.5): #add the foreign object with 50% probability (inefficient, but retained to precisely replicate the paper results)
        DoublesList.append(inst) #add instance to list of objects with two foreign objects
        Placed = False
        while(Placed == False): #find a suitable location: partially overlapping the object but not the first foreign object
            #Pick a random location
            pz = np.random.randint(24,102)
            py = np.random.randint(24,102)
            px = np.random.randint(24,102)
            
            #Check if the second foreign object is (partially) included in the main object
            while(not np.any(np.array([Object[pz-8:pz+8, py-8:py+8, px-8:px+8] == 1]) & np.array([ell2 == 1]))):
                pz = np.random.randint(24,102)
                py = np.random.randint(24,102)
                px = np.random.randint(24,102)

            #If the second foreign object does not overlap the first, place it
            FO2 = np.copy(ell2)
            FO2[FO2 > 0] = 3
            if(np.all(Object[pz-8:pz+8, py-8:py+8, px-8:px+8][FO2>0] != 2)):
                Object[pz-8:pz+8, py-8:py+8, px-8:px+8][FO2>0] = FO2[FO2 > 0]
                Placed = True

    #Save the objects in the training and test folders and log the files with two foreign objects
    if (inst < 100):
        tifffile.imsave(outputPath + 'Objects3DTrain/' + str(inst).zfill(3) + 'instObj3D.tiff', Object.astype(np.uint8))
        if (inst == 99):
            with open("DoublesListTrain.txt", "wb") as fp:
                pickle.dump(DoublesList, fp)
            DoublesList = []
    else:
        tifffile.imsave(outputPath + 'Objects3DTest/' + str(inst-100).zfill(3) + 'instObj3D.tiff', Object.astype(np.uint8))
        if (inst == 499):
            with open("DoublesListTest.txt", "wb") as fp:
                pickle.dump(DoublesList, fp)
