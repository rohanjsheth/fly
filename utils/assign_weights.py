import numpy as np


def assign_weights(neurons, edges, rng):
    edges["neural_weight"] = rng.normal(scale=np.sqrt(2/neurons["fan_in"].reindex(edges["target"]).values))