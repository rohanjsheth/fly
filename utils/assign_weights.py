import numpy as np


def weight_scale(neurons, edges):
    return np.sqrt(2 / neurons["fan_in"].reindex(edges["target"]).values)


def assign_weights(neurons, edges, rng):
    edges["neural_weight"] = rng.normal(scale=weight_scale(neurons, edges))
