import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import jaxspec

import pytest

from jaxspec.data import ObsConfiguration
from jaxspec.model.additive import Powerlaw,Diskbb
from jaxspec.model.multiplicative import Tbabs


# load data
def load_data_jaxpec(grouped_spectrum):
    """Load data from a grouped spectrum."""
    obs = ObsConfiguration.from_pha_file(grouped_spectrum,low_energy=0.2, high_energy=12.0)
    return obs

# create model
def create_model_jaxspec():
    """Create a model."""
    model = Tbabs() * Powerlaw() #+ Diskbb())
    return model

# evaluate model
def evaluate_model_jaxspec(model, params):
    """Evaluate the model."""
    energies = jnp.geomspace(0.2, 12.0, 1000)
    return model.photon_flux(params,energies[:-1],energies[1:],n_points=30).block_until_ready()

# benchmark the code
def test_jaxspec(benchmark):
    params = {"powerlaw_1_alpha":2.,
    "powerlaw_1_norm": 1e-3,
     "tbabs_1_nh": 0.5}
    # run a first time 
    model = create_model_jaxspec()
    _ = evaluate_model_jaxspec(model,params)
    result = benchmark(evaluate_model_jaxspec,model,params=params)
