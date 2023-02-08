#Creation of spectral projections from material projections for demonstration of the workflow

#This scripts converts material projections into (sufficiently) realistic spectral projections
#The results are the (spectral) X-ray projections for the training and test sets
#The code assumes that the PhantomGeneratorTrainandTest.py and PhantomProjectorTrainandTest.py have been carried out first to produce the material projections
# Code contains possibility to change spectral characteristics (multiple spectral bins are supported but requires changes in workflow and training scripts)

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import csv
import numpy as np
import operator
import os
#import physdata.xray #(optional package)
import scipy.interpolate
import tifffile
from ElementaryData import *

#Genertic settings (for spectral data settings, see the Spectral class
NIST = False                                            #If True, attenuation data is taken from NIST website, otherwise from local copy
path = '../../../data/Numerical/'                       #Output folder
FFAverages = 20                                         #Number of flatfield images to correct the data with
Noise = True
InstanceBegin = 0
InstanceEnd = 500


### Import operations to speed up computations 
import ctypes
lib = ctypes.CDLL('./operations.so')
aslong = ctypes.c_uint64
asfloat = ctypes.c_float
asuint = ctypes.c_uint
cfloatp = ctypes.POINTER(ctypes.c_float)
def asfloatp(arr):
    return arr.ctypes.data_as(cfloatp)
# Try to set number of threads to number of physical cores
try:
    import psutil
    ncpu = psutil.cpu_count(logical=False)
    naff = len(psutil.Process().cpu_affinity())
    if naff < ncpu:
        ncpu = naff
    lib.set_threads(asuint(ncpu))
except ImportError:
    pass
###


#Converts a given material name to its atomic number
def elementToAtomic(materialname):
    Candidates = [x for x in ElementaryData if x[1] == materialname]
    if not Candidates:
        return 0
    else:
        return next(x for x in ElementaryData if x[1] == materialname)[0]
        

class Spectra(object):
    def __init__(self):
        self.Labels = [(0, 'Void')]     #List of labels assigned to the materials (initially only void)
        self.SourceSpectrum = None      #Scipy function that determines the photon source intensity given an energy (constructed using linear interpolation)
        self.AttenuationSpectra = []    #List containing all relevant attenuation spectrum functions (constructed on the fly)
                                        #Contains tuples of the form (AtomicNumber, Name, EnergyData, AttenuationData, InterpolationFunction) 

    #Initialize source spectrum from file (in files the entries are separated by commas)
    def InitializeSourceSpectrum(self, path = None):
        if path is not None:
            #Read the given file
            with open(path) as f:
                lines = f.readlines()
                lines = [line.replace(',', '.') for line in lines]  #Comma decimal marker -> dot
                lines.insert(0, 'keV;photon_density\n')             #Manually add column labels
            dialect = csv.Sniffer().sniff(lines[0], delimiters=';')
            reader = csv.DictReader(lines, dialect=dialect)
            energies = np.empty(len(lines) - 1)
            photon_dens = np.empty(len(lines) - 1)
            for i, row in enumerate(reader):
                energies[i] = float(row['keV']) 
                photon_dens[i] = float(row['photon_density'])       #[mm^(-2)]
        else:
            energies = [0,1000]
            photon_dens = [1,1]
        #Create an interpolation function for the given energt-photon density tuples
        source_spectrum = scipy.interpolate.interp1d(energies, photon_dens)
        self.SourceSpectrum = source_spectrum

    #Initialize the attenuation spectra associated to the material labels
    def collectAttenuationSpectra(self, Rootdatapath):
        for mat in self.Labels:
            if(mat[0] != 0 and mat[1] != "Void"): #Exclude Voids                
                if (mat[1] in [i[1] for i in ElementaryData]):                          #Elementary material
                    AtNo = elementToAtomic(mat[1])
                    if (AtNo > 0):
                        attData = self.getAttenuationSpectrum(AtNo, Rootdatapath)
                        self.AttenuationSpectra.append((mat[0],)+(mat[1],) + attData)
                else:                                                                   #Mixture material
                    attData = self.getAttenuationSpectrum(mat[1], Rootdatapath)                                                  
                    self.AttenuationSpectra.append((mat[0],)+(mat[1],) + attData)
            elif(mat[0] != 0 and mat[1] == "Void"):
                #Make the zero attenuation spectrum
                x, y = np.arange(0,100), np.zeros(100)
                spectrum = scipy.interpolate.interp1d(x, y)
                self.AttenuationSpectra.append((mat[0],)+('Void',) + (x,y,spectrum))

        self.AttenuationSpectra.sort(key = operator.itemgetter(0)) #Keep sorted on atomic number  

    #Add labels for material index (value) to list of labels
    def updateLabel(self, value, labeling):
        if isinstance(labeling, int):
            label = ElementaryData[labeling][1]
        else:
            label = labeling
        self.Labels.append((value, label))

    #Get attenuation spectrum for a given material number
    def getAttenuationSpectrum(self, materialno, rootdatapath):
        if NIST == True:    #Take data from online NIST website
            data = np.array(physdata.xray.fetch_coefficients(materialno))
        else:               #Take data from local copy of the NIST website
            data = self.fetchCoefficientsCustom(materialno, rootdatapath)
        data[:, 0] *= 1000  #Convert from MV to kV
       
        #Create (linear) interpolation function
        x, y = data[:, 0], data[:, 1]
        spectrum = scipy.interpolate.interp1d(x, y)

        return data[:,0], data[:,1], spectrum

    #Get attenuation spectrum data from local copy (in case NIST is unreachable)
    def fetchCoefficientsCustom(self, arg, Rootdatapath):
        #Helper function for finding files with prefixes
        def findPrefixFile(prefix, path):
            for i in [f for f in sorted(os.listdir(path)) if not f.endswith('~')]:
                if os.path.isfile(os.path.join(path,i)) and prefix in i:
                    return i
        #Find the data file
        if type(arg) is int or type(arg) is np.uint8:
            path = Rootdatapath + 'DataElemental/'
            pathtofile = path + findPrefixFile(str(arg).zfill(2) +'-', path)
        elif type(arg) is str:
            path = Rootdatapath + 'DataMixture/'
            pathtofile = path + findPrefixFile(arg + ' - ', path)
        #Open the data file and read and parse contents
        with open(pathtofile) as f:
            content = f.readlines()
        content = np.array(content)
        content2 = np.zeros(((content.size),3))
        for ind, i in enumerate(content):
            dataline = i.split(' ')
            dataline = [x for x in dataline if (x != '' and x != '\n')]
            content2[ind,:] = [float(x) for x in dataline[-3:]]
        #Return nx3 table
        return content2


class ProjectionGenerator(object):
    def __init__(self):
        #Material projections
        self.PathToProjs = '../../../data/Numerical/MaterialProjectionsTrain/'                                          #Path to folder with material projections

        self.sizey = 128                                                                                                #Size of the material projections y
        self.sizex = 128                                                                                                #Size of the material projections x
        self.Materials = 3                                                                                              #Number of materials, excluding empty area

        #Source spectrum
        self.fileSource = '../../../data/Numerical/SourceSpectra/Radiology_Source_Thungsten_100kV_NoFilter.csv'         #Path to source spectrum
        self.exposureTime = 0.002                                                                                       #Exposure time in seconds
        self.scaling = 17805999                                                                                         #Look in spectrumscaling.txt for the right values
        self.current = 10                                                                                               #Current/power of the machine (just used as a multiplication factor to decrease zero counts (after noise)        
        self.DetPixelSize = 0.15
        self.SourceSpectrumScaling = self.scaling*self.exposureTime*self.DetPixelSize*self.DetPixelSize*self.current                                                 

        #Material spectra
        self.Rootdatapath = '../../../data/Numerical/NIST/RawData/'                                                     #Location of the NIST data
        self.AttSpectraInfo = [[1,'tissue',None],[2,'bone',None],[3,'bone',None]]                                       #List of material properties in the following format [identifier, atomnumber/NIST material name, file/array]
        self.Materials = len(self.AttSpectraInfo)                                                                       #  one of the latter two has to be None, otherwise the first argument will go first
                                                                                                                
        #Energy bin partition
        self.EnergyBounds = None                                                                                        #Energy partition, if empty then parition will be made with values below (with equal spacing)
        self.MinEnergy = 15                                                                                             #Lower bound energy partition
        self.MaxEnergy = 90                                                                                             #Upper bound energy partition
        self.EnergyBins = 1                                                                                             #Number of energy bins

        #Projection computation
        self.VoxelSize = 0.1                                                                                            #cm/pixel
        self.IntegralBins = 1000      

        #Select the energy range and precision over all projections
        if not self.EnergyBounds:
            self.EnergyBounds = np.linspace(self.MinEnergy, self.MaxEnergy, num = self.EnergyBins+1)
        else:
            self.EnergyBins = len(self.EnergyBounds)-1
        print("Energy binning:", self.EnergyBounds)
        
        #Setup spectra
        self.Sp = Spectra()
        
        #Initialize source spectrum array (for flatfield and projection computations)
        self.Sp.InitializeSourceSpectrum(path = self.fileSource)


    #Create the flatfield image for all projections
    def MakeFlatfieldProjection(self):
    
        print("Creating the flatfield image...")
        FFclean = np.zeros((self.EnergyBins, self.sizey, self.sizex), dtype=np.float32)

        for i in range(0,len(self.EnergyBounds)-1):
            print("Bin [", self.EnergyBounds[i], ",", self.EnergyBounds[i+1], "]")
            #Make full integral energy projection - using midpoint rule
            EMin = self.EnergyBounds[i]
            EMax = self.EnergyBounds[i+1]
            energies = np.linspace(EMin, EMax, num = self.IntegralBins+1)
            binWidth = (EMax - EMin)/self.IntegralBins
            #Compute material summation in the exponent 
            for e in range(0, self.IntegralBins):
                FFclean[i,:,:] += binWidth*self.Sp.SourceSpectrum((energies[e]+energies[e+1])*0.5)*self.SourceSpectrumScaling        
        print("Flatfield created!")
        
        return FFclean

    #Create the spectral projection for a given instance and projection angle
    def MakeSpectralProjection(self, Instance, Angle):

        #Prepare attenuation spectrum arrays
        for idx, item in enumerate(self.AttSpectraInfo):
            if item[0] <= self.Materials:
                self.Sp.updateLabel(item[0], item[1])
            else:
                print("No material identifier ", item[0], "in the data")

        #Collect the attenuation spectra of all involved materials
        self.Sp.collectAttenuationSpectra(self.Rootdatapath)

        #Create projection for all instances
        SpecProj = np.zeros((self.EnergyBins, self.sizey, self.sizex), dtype=np.float32)

        MatProjsInst = tifffile.imread(self.PathToProjs + '/Instance' + str(Instance).zfill(3) + '/Instance' + str(Instance).zfill(3) + 'Angle' + str(Angle).zfill(5) + '.tiff')[0:self.Materials,:,:]

        for i in range(0,len(self.EnergyBounds)-1):
            print("Bin [", self.EnergyBounds[i], ",", self.EnergyBounds[i+1], "]")
            
            #Make full integral energy projection - using midpoint rule
            EMin = self.EnergyBounds[i]
            EMax = self.EnergyBounds[i+1]
            energies = np.linspace(EMin, EMax, num = self.IntegralBins+1)
            binWidth = (EMax - EMin)/self.IntegralBins

            #Compute material summation in the exponent 
            atts = []
            for m in range(1, self.Materials+1): 
                atts.append([x[4] for x in self.Sp.AttenuationSpectra if x[0] == m])
            TotalProjection = np.zeros((self.sizey, self.sizex), dtype=np.float32)
            for e in range(0, self.IntegralBins):
                lib.zero(asfloatp(TotalProjection.ravel()), aslong(TotalProjection.size))
                for m in range(1, self.Materials+1): 
                    #Find the spectrum related to material m
                    Sp = atts[m-1]
                    att = Sp[0]((energies[e]+energies[e+1])*0.5)*self.VoxelSize
                    lib.cplusab(asfloatp(MatProjsInst[m-1].ravel()),asfloat(att), asfloatp(TotalProjection.ravel()), aslong(TotalProjection.size))
                lib.cplusexpab(asfloatp(TotalProjection.ravel()),asfloat(binWidth*self.Sp.SourceSpectrum((energies[e]+energies[e+1])*0.5)*self.SourceSpectrumScaling), asfloatp(SpecProj[i]), aslong(TotalProjection.size))

        return SpecProj   




def main():
    PG = ProjectionGenerator()

    #Create the flatfield image
    FFclean = PG.MakeFlatfieldProjection()

    #Create output directory
    os.makedirs(path + '/ProjectionDataTrain/', exist_ok=True)
    os.makedirs(path + '/ProjectionDataTest/', exist_ok=True)

    #Loop over all phantom instances
    for Instance in range(InstanceBegin,InstanceEnd):

        #Set correct save folder, set random seed (for when noise is applied) and create the output directory for the spectral projections
        if Instance < 100:
            fullpath = path + '/ProjectionDataTrain'
            np.random.seed(123+Instance)
            os.makedirs(fullpath + '/Instance' + str(Instance).zfill(3) + '/', exist_ok=True)
        else:
            PG.PathToProjs = '../../../data/Numerical/MaterialProjectionsTest/'
            fullpath = path + '/ProjectionDataTest'
            np.random.seed(123+Instance-100)
            os.makedirs(fullpath + '/Instance' + str(Instance-100).zfill(3) + '/', exist_ok=True)

        #Apply noise to the flatfield image
        if(Noise):
            FFSum = np.random.poisson(FFclean)
            for h in range(1, FFAverages):
                FFSum += np.random.poisson(FFclean)
            FF = FFSum/float(FFAverages)

        #List the projeciton angles
        if Instance < 100:
            Angles = range(0, len(sorted(os.listdir(PG.PathToProjs + 'Instance' + str(Instance).zfill(3) + '/'))))
        else:
            Angles = [int(x[-9:-5]) for x in sorted(os.listdir(PG.PathToProjs + 'Instance' + str(Instance-100).zfill(3) + '/'))]
        #Loop over the angles
        for ang in Angles:
            print("--- Object instance", Instance, "Angle", ang, "---")
            #Retrieve the spectral projection
            if Instance < 100:
                Res = PG.MakeSpectralProjection(Instance, ang)
            else:
                Res = PG.MakeSpectralProjection(Instance-100, ang)
            #Apply noise to the spectral projection
            if(Noise):
                Res = np.random.poisson(Res)
            #Apply the flatfield correction
            Proj = np.log(FF[:,:,:]/Res[:,:,:])
            #Remove possibly infinite values resulting from flatfield correction
            Proj[~np.isfinite(Proj)] = 0  
            #Save the material projections to the selected folders
            if Instance < 100:
                tifffile.imsave(fullpath + '/Instance' + str(Instance).zfill(3) + '/Instance' + str(Instance).zfill(3) + 'Angle{:05d}Data.tiff'.format(ang), Proj.astype(np.float32))
            else:
                tifffile.imsave(fullpath + '/Instance' + str(Instance-100).zfill(3) + '/Instance' + str(Instance-100).zfill(3) + 'Angle{:05d}Data.tiff'.format(ang), Proj.astype(np.float32))
            print("Projection created and saved.")
        
        #Log the settings for this instance
        with open(fullpath + '/materialsettingsInst' + str(Instance) + '.txt', 'w') as f:
            f.write("Noise: %s\n" % Noise) 
            f.write("FFAverages: %s\n" % FFAverages) 
            f.write("MinEnergy: %s\n" % PG.MinEnergy) 
            f.write("MaxEnergy: %s\n" % PG.MaxEnergy)
            f.write("EnergyBins: %s\n" % PG.EnergyBins)
            f.write("IntegralBins: %s\n" % PG.IntegralBins) 
            f.write("ExposureTime: %s\n" % PG.exposureTime)
            f.write("FileSourceSpectrum: %s\n" % PG.fileSource)
            f.write("Scaling: %s\n" % PG.scaling)
            f.write("Machinecurrent: %s\n" % PG.current)
            f.write("DetectorPixelSize: %s\n" % PG.DetPixelSize)
            f.write("VoxelSize: %s\n" % PG.VoxelSize)
            for item in PG.AttSpectraInfo:
                f.write("%s\n" % item)
            
if __name__ == "__main__":
    main()
