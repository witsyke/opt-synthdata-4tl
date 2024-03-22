# Data Details

## Overview
This folder provides all the real-world data for the two rotifers-algae experiments and the necessary scripts to generate the synthetic datasets from the SAR model from Rosenbaum et al. (https://doi.org/10.3389/fevo.2018.00234).

## File Structure
- The real-world/C1.csv contains the original rotifers-algae data from Blasius et al. (2019).
- We define two subsets (see real-world folder) with coherent and incoherent data according to analysis by Blasius et al.
- Running the rosenbaum.py generates the synthetic datasets from the SAR models as described in the appendix of the paper.
- We provide an exemplary synthetic datasets in /synthetic

## Data Sources
The real-world datasets is provided in Blasius, B., Rudolf, L., Weithoff, G., Gaedke, U., & Fussmann, G. F. (2019). Long-term cyclic persistence in an experimental predator–prey system. Nature, 577(7789), 226–230.

## Data Preprocessing
We interpolated the incoherent subset due to a missing datapoint in the original dataset by Blasius et al.