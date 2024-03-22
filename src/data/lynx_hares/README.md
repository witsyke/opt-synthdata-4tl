# Data Details

## Overview
This folder provides all the real-world data for the lynx-hares experiments and the necessary scripts to generate the synthetic datasets from the LV model from Lotka (1920)/Volterra (1926).

## File Structure
- The real-world data is provided in real-world/lynx-hares.csv
- You can inspect the real-world data in plot.ipynb
- We generate the synthetic datasets from a fitted Lotka-Volterra ODE-model in lotka_volterra.py
- We provide an exemplary synthetic datasets in /synthetic

## Data Sources
We obtain the data from 'http://people.whitman.edu/~hundledr/courses/M250F03/LynxHare.txt', which corresponds to the dataset in MacLulich (1937) - Fluctuations in the numbers of the varying hare (Lepus americanus).

## Data Preprocessing
Details on pre-processing steps are described in the appendix of the paper.