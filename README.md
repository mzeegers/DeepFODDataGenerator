# <a href="https://www.sciencedirect.com/science/article/pii/S0957417422010429" style="color: black;">DeepFODDataGenerator</a>: A workflow for generating deep learning data through X-ray tomography for X-ray based object detection

   <p align="center">
   <img src="./images/Workflowv3Applied_simplified.svg">
    </p>
    


## Example results:

Below are sample results of comparisons of two quality measures between different training approaches with data generated by the workflow:
   <p align="center">
   <img src="./images/Results_MSDUNET_5Avgs_AvgClassAcc_shaded.png">
   </p>
   <p align="center">
   <img src="./images/Results_MSDUNET_5Avgs_FPrate_shaded.png">
   </p>
   

## References

The algorithms and routines implemented in this python package are described in following [paper](https://www.sciencedirect.com/science/article/pii/S0957417422010429) published in Expert Systems with Applications. If you use (parts of) this code in a publication, we would appreciate it if you would refer to:

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
