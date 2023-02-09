# <a href="https://www.sciencedirect.com/science/article/pii/S0957417422010429" style="color: black;">DeepFODDataGenerator</a>: A workflow for generating deep learning data for X-ray based object detection

   <p align="center">
   <img src="./images/Workflowv3Applied_simplified.svg">
    </p>
    
## Introduction

DeepFODDataGenerator is a Python package for easy generation of training data to apply deep learning for detection of (foreign) objects detection on X-ray projections. The generator follows a workflow of creating reconstructions from CT scans of training objects, and subsequently applying segmentation and virtual projections. The results are ground truth locations of the foreign objects on the X-ray projections from the CT scan, without having to manually annotate each of these.  

This package provides scripts for applying this workflow for any X-ray object detection method, given that there are CT objects of training objects available. Additionally, the package provides scripts for complete reproduction of the experiments in the associated paper titled '[A tomographic workflow to enable deep learning for X-ray based foreign object detection](https://www.sciencedirect.com/science/article/pii/S0957417422010429)', in which the workflow is demonstrated and analyzed through a series of experiments with laboratory CT data and numerical data.  


## Requirements

Running the scripts requires a number of major and minor packages:

**Major**:
1. ASTRA Toolbox:
https://github.com/astra-toolbox/astra-toolbox
2. MSDNet:
https://github.com/dmpelt/msdnet
3. PyTorch:
https://pytorch.org/

Minor (but essential):
csv, cv2 (OpenCV), matplotlib, numpy, pickle, scipy, sklearn, tiffile, tqdm

*Optional*:
cudatoolkit (recommended), physdata.xray, pyqtgraph (useful for plotting and examining intermediate results),

The code has been tested with Python version 3.8.1 on Fedora 36, with Intel(R) Core(TM) i7-7700K CPU, GeForce GTX 1070 GPU, CUDA version 11.8 and CUDA toolkit 10.1.243.

Also, for running the workflow with experimental data, it is required to download the X-ray CT scans dataset available at [Zenodo](https://zenodo.org/record/5866228) and store these in the data/CTData folder.

It is recommended to have storage space of at least 500 GB to run the full experimental reproduction of the results in the paper. For your own experiments derived from this code, this amount of storage space is of course not needed.

## Scripts

The script folder is organized in the following manner:

```
├── scripts
    ├── experimental
    │   ├── generation
    │   │   └────────── ReconstructAndProject.py
    |   ├── training
    |   │   ├── MSD
    |   │   │   ├── FewFOs
    |   │   │   │   └── WorkflowExperimentalDataMSD_train_segm.py
    |   │   │   ├── FewFOsOneRadiograph
    |   │   │   │   └── WorkflowExperimentalDataOneRadiographMSD_train_segm.py
    |   │   │   ├── ManyFOs
    |   │   │   │   └── WorkflowExperimentalDataManyFOMSD_train_segm.py
    |   │   │   └── MixedFOs
    |   │   │       └── WorkflowExperimentalDataMixedMSD_train_segm.py
    |   │   └── UNet
    |   │       ├── FewFOs
    |   │       │   └── WorkflowExperimentalDataUNet_train_segm.py
    |   │       ├── FewFOsOneRadiograph
    |   │       │   └── WorkflowExperimentalDataOneRadiographUNet_train_segm.py
    |   │       ├── ManyFOs
    |   │       │   └── WorkflowExperimentalDataManyFOUNet_train_segm.py
    |   │       └── MixedFOs
    |   │           └── WorkflowExperimentalDataMixedUNet_train_segm.py
    |   ├── testing
    |   |   ├────────── WorkflowExperimentalDataMSD_Test.py
    |   |   └────────── WorkflowExperimentalDataUNet_Test.py
    |   └── plotting
    |       ├────────── PlotAllResultsDiffFOamounts.py
    |       └────────── PlotAllResultsManyvsOneProj.py
    └── numerical
        ├── generation
        |   ├────────── PhantomGeneratorTrainandTest.py
        |   ├────────── PhantomProjectorTrainandTest.py
        |   ├────────── SpectralDataGeneratorTrainandTest.py
        |   ├────────── ReconstructAndProject.py
        |   └────────── CompareSegmentationsandGT.py
        ├── training
        |   ├── MSD
        |   |   ├── FewFOs
        |   |   |   └── WorkflowNumericalDataMSD_train_segm.py
        |   |   └── FewFOsOneRadiograph
        |   |       └── WorkflowNumericalDataOneRadiographMSD_train_segm.py
        |   └── UNet
        |       ├── FewFOs
        |       |   └── WorkflowNumericalDataUNet_train_segm.py
        |       └── FewFOsOneRadiograph
        |           └── WorkflowNumericalDataOneRadiographUNet_train_segm.py
        ├── testing
        |   ├────────── WorkflowNumericalDataMSD_Test.py
        |   └────────── WorkflowNumericalDataUNet_Test.py
        └── plotting
            └────────── PlotAllResultsManyvsOneProj.py
```    

For the training of the networks there are scripting files available that allows for scheduling the training procedures on the GPU (named ScripterCUDA*coreindex*).

To reproduce the various results in the [paper](https://www.sciencedirect.com/science/article/pii/S0957417422010429), follow the following scripts:
- **Section 4.5:** Run all scripts in the scripts/expermental/ folder in the above order (with multiple network trainings with the number of objects and runs as described in this section).
- **Section 4.6**: Same as above.
- **Section 4.7**: Repeat the experiments for section 4.5 with different thresholds in the ReconstructAndProject.py script.
- **Section 4.8**: Run all scripts in the scripts/numerical/ folder in the above order (with multiple network trainings with the number of objects and runs as described in this section).

The resulting plots (see example below) will be located /results folder.

## Example results:

Below are sample results of comparisons of two quality measures between different training approaches with data generated by the workflow:
   <p align="center">
   <img src="./images/Results_MSDUNET_5Avgs_AvgClassAcc_shaded.png" style="width: 45%">
   <img src="./images/Results_MSDUNET_5Avgs_FPrate_shaded.png" style="width: 45%">
   </p>
   

## References

The algorithms and routines implemented in this Python package are described in following [paper](https://www.sciencedirect.com/science/article/pii/S0957417422010429) published in Expert Systems with Applications. If you use (parts of) this code in a publication, we would appreciate it if you would refer to:

```
@article{,
  title={A tomographic workflow to enable deep learning for X-ray based foreign object detection},
  author={Zeegers, Math{\'e} T and van Leeuwen, Tristan and Pelt, Dani{\"e}l M and Coban, Sophia Bethany and van Liere, Robert and Batenburg, Kees Joost},
  journal={Expert Systems with Applications},
  volume={206},
  pages={117768},
  year={2022},
  publisher={Elsevier}
}
```
The preprint can be found [here](https://arxiv.org/abs/2201.12184).

The X-ray CT scans dataset required to run the experimental data scripts can be found at [Zenodo](https://zenodo.org/record/5866228).


## Authors

Code written by:
- Mathé Zeegers (m [dot] t [dot] zeegers [at] cwi [dot] nl).

The MSD and UNet training scripts contain elements of MSD code (https://github.com/dmpelt/msdnet) by Daniël Pelt and PyTorch UNet code (https://github.com/usuyama/pytorch-unet) by Naoto Usuyama respectively.
