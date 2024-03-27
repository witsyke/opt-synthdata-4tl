# Code Repository - Optimizing ODE-derived Synthetic Data for Transfer Learning in Dynamical Biological Systems

This code repository provides the necessary tools to reproduce the findings from Zabbarov et al. on optimizing synthetic datasets characteristics for a simulation-based transfer learning approach to predicting dynamical biological systems.

![Overview of Experimental Setup](figures/visual_abstract.png)

## Installation

To run the code, please setup a new environment and install the required dependencies, i.e. using pip:
```
pip install -r dev_requirements.txt
```
Please call each script from the root directory of this repository.

## Structure

- Datasets: The datasets for the rotifers-algae, lynx-hares and COVID-19 experiments are provided in `src/data/`. Here, we also provide the scripts for generating syntehtic datasets from the calibrated ODE models. For further details, see the README.mds in the respective folders.
- Transfer Learning and Deep Learning: The code to run the multivariate transfer learning experiments and the DL baseliens for each biological system is contained in `src/experiments/julian/`.
- DL models: The small and large architecture DL models are defined as plugins for SimbaML in `src/models`.
- ODE calibration: The code used to calibrate the ODEs for each each biological system is contained in `src/experiments/simon/`.
