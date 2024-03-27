import random
import os

from tqdm import tqdm
import numpy as np

from simba_ml.simulation import system_model
from simba_ml.simulation import species
from simba_ml.simulation import noisers
from simba_ml.simulation import generators
from simba_ml.simulation import distributions
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module
from simba_ml.simulation import derivative_noiser
from simba_ml.simulation import constraints
from simba_ml.simulation import random_generator

NAME = "lotka-volterra"
seeds = [10, 42, 17, 93, 97]

# choice of kinetic parameters and initial conditions based on fitting from
# https://mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html

sd = [0.064, 0.004, 0.092, 0.004]
p_init = [0.545, 0.028, 0.803, 0.024]

initial_conditions_config = {
    "IC0": (distributions.Constant(20), distributions.Constant(34)),
    "IC1": (distributions.ContinuousUniformDistribution(18, 22),
                distributions.ContinuousUniformDistribution(30.6, 37.4)),
    "IC2": (distributions.ContinuousUniformDistribution(16, 24),
                distributions.ContinuousUniformDistribution(27.2, 40.8)),
    "IC3": (distributions.ContinuousUniformDistribution(14, 26),
                distributions.ContinuousUniformDistribution(23.8, 44.2)),
    }

kinetic_parameters_config = {"P0": (distributions.Constant(p_init[0]),
                                    distributions.Constant(p_init[1]),
                                    distributions.Constant(p_init[2]),
                                    distributions.Constant(p_init[3])),
                             "P1": (distributions.ContinuousUniformDistribution(round(p_init[0]-(1/2)*sd[0], 3), round(p_init[0]+(1/2)*sd[0], 3)),
                                    distributions.ContinuousUniformDistribution(round(p_init[1]-(1/2)*sd[1], 3), round(p_init[1]+(1/2)*sd[1], 3)),
                                    distributions.ContinuousUniformDistribution(round(p_init[2]-(1/2)*sd[2], 3), round(p_init[2]+(1/2)*sd[2], 3)),
                                    distributions.ContinuousUniformDistribution(round(p_init[3]-(1/2)*sd[3], 3), round(p_init[3]+(1/2)*sd[3], 3))),
                             "P2": (distributions.ContinuousUniformDistribution(round(p_init[0]-(2/2)*sd[0], 3), round(p_init[0]+(2/2)*sd[0], 3)),
                                    distributions.ContinuousUniformDistribution(round(p_init[1]-(2/2)*sd[1], 3), round(p_init[1]+(2/2)*sd[1], 3)),
                                    distributions.ContinuousUniformDistribution(round(p_init[2]-(2/2)*sd[2], 3), round(p_init[2]+(2/2)*sd[2], 3)),
                                    distributions.ContinuousUniformDistribution(round(p_init[3]-(2/2)*sd[3], 3), round(p_init[3]+(2/2)*sd[3], 3))),
                             "P3": (distributions.ContinuousUniformDistribution(round(p_init[0]-(3/2)*sd[0], 3), round(p_init[0]+(3/2)*sd[0], 3)),
                                    distributions.ContinuousUniformDistribution(round(p_init[1]-(3/2)*sd[1], 3), round(p_init[1]+(3/2)*sd[1], 3)),
                                    distributions.ContinuousUniformDistribution(round(p_init[2]-(3/2)*sd[2], 3), round(p_init[2]+(3/2)*sd[2], 3)),
                                    distributions.ContinuousUniformDistribution(round(p_init[3]-(3/2)*sd[3], 3), round(p_init[3]+(3/2)*sd[3], 3)))}

size_config = {
    "TS1": 1,
    "TS10": 10,
    "TS100": 100,
    "TS1000": 1000
    }
                             
for seed in seeds:
    for size_key, size_value in tqdm(size_config.items(), leave=False, desc="Size", position=0):
        for ic_key, ic_value in tqdm(initial_conditions_config.items(), leave=False, desc="Initial Conditions", position=1):
            for kp_key, kp_value in tqdm(kinetic_parameters_config.items(), leave=False, desc="Kinetic Parameters", position=2):
                random_generator.set_seed(seed)
                specieses = [
                    species.Species("Lynx", ic_value[0], min_value=0),
                    species.Species("Hare", ic_value[1], min_value=0),
                ]
                kinetic_parameters = {
                    "alpha": kinetic_parameters_module.ConstantKineticParameter(kp_value[0]), # Reproduction rate of prey
                    "beta": kinetic_parameters_module.ConstantKineticParameter(kp_value[1]), # Mortality rate of predator per prey
                    "gamma": kinetic_parameters_module.ConstantKineticParameter(kp_value[2]), # Mortality rate of predator
                    "delta": kinetic_parameters_module.ConstantKineticParameter(kp_value[3]) # Reproduction rate of predator per prey
                }
                def deriv(_t, y, kinetic_parameters):
                    predator, prey = y
                    dprey_dt = prey * (kinetic_parameters["alpha"] - kinetic_parameters["beta"] * predator)
                    dpredator_dt = predator * (kinetic_parameters["delta"] * prey - kinetic_parameters["gamma"])
                    return dpredator_dt, dprey_dt
                sm = system_model.SystemModel(
                    NAME,
                    specieses,
                    kinetic_parameters,
                    deriv=deriv,
                    timestamps=distributions.Constant(100),
                    solver_method="LSODA",
                    atol=1e-8,
                    rtol=1e-5,
                )
                sm = constraints.SpeciesValueTruncator(sm)
                print("Generating:" + f"src/data/lynx-hares/synthetic_fitted/{size_key}_{ic_key}_{kp_key}_S{seed}")
                generators.TimeSeriesGenerator(sm).generate_csvs(size_value, os.getcwd() + f"/src/data/lynx-hares/synthetic/{size_key}_{ic_key}_{kp_key}_S{seed}")