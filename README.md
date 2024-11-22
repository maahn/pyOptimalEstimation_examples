[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maahn/pyOptimalEstimation_examples/master?filepath=Index.ipynb)

# Optimal Estimation Retrievals and Their Uncertainties: What Every Atmospheric Scientist Should Know 
# Supplemental material

Maahn, M., D. D. Turner, U. Löhnert, D. J. Posselt, K. Ebell, G. G. Mace, and J. M. Comstock, 2020: Optimal Estimation Retrievals and Their Uncertainties: What Every Atmospheric Scientist Should Know. Bull. Amer. Meteor. Soc., doi:https://doi.org/10.1175/BAMS-D-19-0027.1

This repository contains examples illustrating the use of the [pyOptimalEstimation library](https://github.com/maahn/pyOptimalEstimation). Two Juptyter Notebooks are provided:

* [Supplement A: Microwave Radiometer Temperature and Humidity Retrieval ](Supplement%20A%20-%20MWR%20retrieval.ipynb)
* [Supplement B: Cloud Radar Drop Size Distribution Retrieval ](Supplement%20B%20-%20DSD%20retrieval.ipynb)

If you are new to Jupyter Notebooks, pelase check out the [official tutorial](https://mybinder.org/v2/gh/ipython/ipython-in-depth/master?filepath=binder/Index.ipynb).


## How to try the examples online
[You can try the examples online in your browser without any local installation on binder by following this link](https://mybinder.org/v2/gh/maahn/pyOptimalEstimation_examples/master?filepath=Index.ipynb). Note that it takes a minute or two to launch the server, that sometimes you have to try it twice, and that the server shuts down when you do not use it for a couple of minutes. Changes are not saved but a modified Notebook can be downloaded via File > Download as > Notebook

## How to try the examples locally
Unless you have pyOptimalEstimation already installed, it is recommended to 

1. Install [Anaconda](https://www.anaconda.com/distribution/#download-section). Version 3.6 or higher is recommended.
2. Download or clone this repository, e.g. with the green `Code` button. 
3. Open a terminal and navigate to the folder of this repository
4. Install the [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) `pyoe_examples` (so it won't mess with your default Python installation) with 
    1. on Linux: `conda env create -f environment_linux.yml`
    2. on Mac OS X: `conda env create -f environment_macosx.yml`
5. Activate the environemnt with `conda activate pyoe_examples`
6. In the terminal, start the Jupyter server with `jupyter notebook` 
7. Your browser should open automatically and you can open one of the provided ipynb files.
8. Make sure to change the kernel to `pyoe_examples` with Kernel > Set_Kernel





