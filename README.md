[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maahn/pyOptimalEstimation_examples/master?filepath=Index.ipynb)

# pyOptimalEstimation_examples

This repository contains Jupyter Notebook examples for the [pyOptimalEstimation library](https://github.com/maahn/pyOptimalEstimation). If you are new to Jupyter Notebooks, pelase check out the [official tutorial](https://mybinder.org/v2/gh/ipython/ipython-in-depth/master?filepath=binder/Index.ipynb).


## How to try the examples online
You can try the examples online in your browser without installing any code [on binder](https://mybinder.org/v2/gh/maahn/pyOptimalEstimation_examples/master?filepath=Index.ipynb).

## How to try the examples locally
Unless you have pyOptimalEstimation already installed, it is recommended to 

1. Install [Anaconda](https://www.anaconda.com/distribution/#download-section). Version 3.6 or higher is recommended.
2. Download or clone this repository, e.g. with the green `Clone or download` button. 
3. Open a terminal and navigate to the folder of this repository
4. Install the [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) `pyoe_examples` (so it won't mess with your default Python installation) with 
    1. on Linux: `conda env create -f environment_linux.yml`
    2. on Mac OS X: `conda env create -f environment_macosx.yml`
5. In the terminal, start the Jupyter server with `jupyter notebook` and open one of the provided ipynb files.
6. Make sure to change the kernel to `pyoe_examples` with Kernel > Set_Kernel




