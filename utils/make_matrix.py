import numpy as np
from scipy.sparse import csr_matrix


def gen_matrix(edges, num_neurons):
    return csr_matrix(
        (edges.neural_weight, (edges.target, edges.source)),
        shape=(num_neurons, num_neurons),
        dtype=np.float32,
    )
