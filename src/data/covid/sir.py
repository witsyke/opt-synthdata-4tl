import random
import numpy as np
from tqdm import tqdm
from simba_ml.simulation import system_model
from simba_ml.simulation import species
from simba_ml.simulation import noisers

from simba_ml.simulation import constraints
from simba_ml.simulation import distributions
from simba_ml.simulation import sparsifier as sparsifier_module
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module
from simba_ml.simulation import constraints
from simba_ml.simulation import derivative_noiser as derivative_noisers
from simba_ml.simulation import generators
from simba_ml.simulation import random_generator

name="sir"

seeds = [10, 42, 17, 93, 97]

initial_conditions_config = {
    "IC0": (distributions.Constant(84400000.0),
            distributions.Constant(19.2),
            distributions.Constant(0)),
    "IC1": (distributions.ContinuousUniformDistribution(0.9*84400000.0, 1.1*84400000.0),
            distributions.ContinuousUniformDistribution(0.9*19.2, 1.1*19.2),
            distributions.Constant(0)),
    "IC2": (distributions.ContinuousUniformDistribution(0.8*84400000.0, 1.2*84400000.0),
            distributions.ContinuousUniformDistribution(0.8*19.2, 1.2*19.2),
            distributions.Constant(0)),
    "IC3": (distributions.ContinuousUniformDistribution(0.7*84400000.0, 1.3*84400000.0),
            distributions.ContinuousUniformDistribution(0.7*19.2, 1.3*19.2),
            distributions.Constant(0)),
}

kinetic_parameters_config = {"P0": (distributions.Constant(0.12),
                                    distributions.Constant(0.41),),
                             "P1": (distributions.ContinuousUniformDistribution(0.10,  0.15),
                                    distributions.ContinuousUniformDistribution(0.365, 0.46),),
                             "P2": (distributions.ContinuousUniformDistribution(0.08,  0.18),
                                    distributions.ContinuousUniformDistribution(0.32,  0.51),),
                             "P3": (distributions.ContinuousUniformDistribution(0.06,  0.21),
                                    distributions.ContinuousUniformDistribution(0.275, 0.56),)
                            }

size_config = {"TS1": 1,
               "TS10": 10, 
               "TS100": 100,
               "TS1000": 1000,}

for seed in tqdm(seeds, leave=False, desc="Seed", position=0):
    for size_key, size_value in tqdm(size_config.items(), leave=False, desc="Size", position=1):
        for ic_key, ic_value in tqdm(initial_conditions_config.items(), leave=False, desc="Initial Conditions", position=2):
            for kp_key, kp_value in tqdm(kinetic_parameters_config.items(), leave=False, desc="Kinetic Parameters", position=3):
                random_generator.set_seed(seed)
                specieses = [
                    species.Species("Suspectible",
                                    ic_value[0],
                                    contained_in_output=True,
                                    min_value=0,),
                    species.Species("Infected",
                                    ic_value[1],
                                    contained_in_output=True,
                                    min_value=0,),
                    species.Species("Recovered",
                                    ic_value[2],
                                    contained_in_output=True,
                                    min_value=0,),
                ]
                kinetic_parameters: dict[str, kinetic_parameters_module.KineticParameter[float]] = {
                    "beta": kinetic_parameters_module.ConstantKineticParameter(
                        kp_value[1]
                    ),
                    "gamma": kinetic_parameters_module.ConstantKineticParameter(
                        kp_value[0]
                    ),
                }
                def deriv(
                    _t: float, y: list[float], arguments: dict[str, float]
                ) -> tuple[float, float, float]:
                    S, I, _ = y
                    N = sum(y)
                    dS_dt = -arguments["beta"] * S * I / N
                    dI_dt = arguments["beta"] * S * I / N - (arguments["gamma"]) * I
                    dR_dt = arguments["gamma"] * I
                    return dS_dt, dI_dt, dR_dt
                sm = constraints.SpeciesValueTruncator(
                    system_model.SystemModel(
                        name,
                        specieses,
                        kinetic_parameters,
                        deriv=deriv,
                        timestamps=distributions.Constant(100),
                    )
                )
                generators.TimeSeriesGenerator(sm).generate_csvs(size_value, os.getcwd() + f"/synthetic/{size_key}_{ic_key}_{kp_key}_S{seed}")