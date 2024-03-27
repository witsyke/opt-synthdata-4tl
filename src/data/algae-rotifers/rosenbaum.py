import os
import math

from tqdm import tqdm

from simba_ml.simulation import system_model
from simba_ml.simulation import species
from simba_ml.simulation import noisers
from simba_ml.simulation import generators
from simba_ml.simulation import distributions
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module
from simba_ml.simulation import derivative_noiser
from simba_ml.simulation import constraints
from simba_ml.simulation import generators
from simba_ml.simulation import random_generator


name = "rosenbaum"

seeds = [10, 42, 17, 93, 97]

mu_S = 10
mu_A = 0.52*10**9
mu_R = 27.92*10**3

ln_mu_ca = 17.68
ln_mu_fa = 1.38
ln_mu_cr = (-11.82)
ln_mu_fr = 0.23

ln_sigma_ca = 0.08
ln_sigma_fa = 0.05
ln_sigma_cr = 0.11
ln_sigma_fr = 0.03

size_config = {"TS1": 1,
               "TS10": 10,
               "TS100": 100,
               "TS1000": 1000,}

initial_conditions_config = {
    "IC0": (distributions.Constant(mu_S),
            distributions.Constant(mu_A),
            distributions.Constant(mu_R)),
    "IC1": (distributions.ContinuousUniformDistribution(0.9*mu_S, 1.1*mu_S),
            distributions.ContinuousUniformDistribution(0.9*mu_A, 1.1*mu_A),
            distributions.ContinuousUniformDistribution(0.9*mu_R, 1.1*mu_R)),
    "IC2": (distributions.ContinuousUniformDistribution(0.8*mu_S, 1.2*mu_S),
            distributions.ContinuousUniformDistribution(0.8*mu_A, 1.2*mu_A),
            distributions.ContinuousUniformDistribution(0.8*mu_R, 1.2*mu_R)),
    "IC3": (distributions.ContinuousUniformDistribution(0.7*mu_S, 1.3*mu_S),
            distributions.ContinuousUniformDistribution(0.7*mu_A, 1.3*mu_A),
            distributions.ContinuousUniformDistribution(0.7*mu_R, 1.3*mu_R)),
}

kinetic_parameters_config = {"P0": (distributions.Constant(0.55),
                                    distributions.Constant(80),
                                    distributions.Constant(math.e ** ln_mu_ca),
                                    distributions.Constant(math.e ** ln_mu_fa),
                                    distributions.Constant(4.3),
                                    distributions.Constant(math.e ** ln_mu_cr),
                                    distributions.Constant(math.e ** ln_mu_fr),
                                    distributions.Constant(7.5 * 10 ** 8),),
                             "P1": (distributions.Constant(0.55),
                                    distributions.Constant(80),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_ca - (1/2)*ln_sigma_ca),  math.e ** (ln_mu_ca + (1/2)*ln_sigma_ca)),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_fa - (1/2)*ln_sigma_fa),  math.e ** (ln_mu_fa + (1/2)*ln_sigma_fa)),
                                    distributions.Constant(4.3),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_cr - (1/2)*ln_sigma_cr),  math.e ** (ln_mu_cr + (1/2)*ln_sigma_cr)),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_fr - (1/2)*ln_sigma_fr),  math.e ** (ln_mu_fr + (1/2)*ln_sigma_fr)),
                                    distributions.Constant(7.5 * 10 ** 8),),
                             "P2": (distributions.Constant(0.55),
                                    distributions.Constant(80),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_ca - (2/2)*ln_sigma_ca),  math.e ** (ln_mu_ca + (2/2)*ln_sigma_ca)),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_fa - (2/2)*ln_sigma_fa),  math.e ** (ln_mu_fa + (2/2)*ln_sigma_fa)),
                                    distributions.Constant(4.3),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_cr - (2/2)*ln_sigma_cr),  math.e ** (ln_mu_cr + (2/2)*ln_sigma_cr)),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_fr - (2/2)*ln_sigma_fr),  math.e ** (ln_mu_fr + (2/2)*ln_sigma_fr)),
                                    distributions.Constant(7.5 * 10 ** 8),),
                             "P3": (distributions.Constant(0.55),
                                    distributions.Constant(80),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_ca - (3/2)*ln_sigma_ca),  math.e ** (ln_mu_ca + (3/2)*ln_sigma_ca)),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_fa - (3/2)*ln_sigma_fa),  math.e ** (ln_mu_fa + (3/2)*ln_sigma_fa)),
                                    distributions.Constant(4.3),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_cr - (3/2)*ln_sigma_cr),  math.e ** (ln_mu_cr + (3/2)*ln_sigma_cr)),
                                    distributions.ContinuousUniformDistribution(math.e ** (ln_mu_fr - (3/2)*ln_sigma_fr),  math.e ** (ln_mu_fr + (3/2)*ln_sigma_fr)),
                                    distributions.Constant(7.5 * 10 ** 8),),
                            }

for seed in tqdm(seeds, leave=False, desc="Seed", position=0):
    for size_key, size_value in tqdm(size_config.items(), leave=False, desc="Size", position=1):
        for ic_key, ic_value in tqdm(initial_conditions_config.items(), leave=False, desc="Initial Conditions", position=2):
            for kp_key, kp_value in tqdm(kinetic_parameters_config.items(), leave=False, desc="Kinetic Parameters", position=3):
                random_generator.set_seed(seed)
                specieses = [
                    species.Species(
                        "nitrogen", ic_value[0], min_value=0
                    ),
                    species.Species(
                        "algae", ic_value[1], min_value=0
                    ),
                    species.Species(
                        "rotifers", ic_value[2], min_value=0
                    ),
                ]

                kinetic_parameters = {
                    "delta": kinetic_parameters_module.ConstantKineticParameter(
                        kp_value[0]
                    ),
                    "S*": kinetic_parameters_module.ConstantKineticParameter(
                        kp_value[1]
                    ),
                    "ca": kinetic_parameters_module.ConstantKineticParameter(
                        kp_value[2]
                    ),
                    "fa": kinetic_parameters_module.ConstantKineticParameter(
                        kp_value[3]
                    ),
                    "ha": kinetic_parameters_module.ConstantKineticParameter(
                        kp_value[4]
                    ),
                    "cr": kinetic_parameters_module.ConstantKineticParameter(
                        kp_value[5]
                    ),
                    "fr": kinetic_parameters_module.ConstantKineticParameter(
                        kp_value[6]
                    ),
                    "hr": kinetic_parameters_module.ConstantKineticParameter(
                        kp_value[7]
                    ),
                }

                def deriv(t, y, kinetic_parameters):
                    S, A, R = y

                    dSdt = (kinetic_parameters["S*"] - S) * kinetic_parameters["delta"] - (1/kinetic_parameters["ca"]) * (kinetic_parameters["fa"] * S / (kinetic_parameters["ha"] + S)) * A
                    dAdt = ((kinetic_parameters["fa"] * S)/(kinetic_parameters["ha"] + S)) * A - (1/kinetic_parameters["cr"]) * ((kinetic_parameters["fr"] * A) / (kinetic_parameters["hr"] + A)) * R - kinetic_parameters["delta"] * A
                    dRdt = (kinetic_parameters["fr"] * A) / (kinetic_parameters["hr"] + A) * R - kinetic_parameters["delta"] * R

                    return [dSdt, dAdt, dRdt]

                sm = system_model.SystemModel(
                    name,
                    specieses,
                    kinetic_parameters,
                    deriv=deriv,
                    timestamps=distributions.Constant(130),
                    atol=1e-8,
                    rtol=1e-5,
                )
                print(os.getcwd() + f"src/data/algae-rotifers/synthetic/{size_key}_{ic_key}_{kp_key}_S{seed}")
                generators.TimeSeriesGenerator(sm).generate_csvs(size_value, os.getcwd() + f"/src/data/algae-rotifers/synthetic/{size_key}_{ic_key}_{kp_key}_S{seed}")