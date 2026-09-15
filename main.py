from pathlib import Path

import numpy as np
import pandas as pd

from utils.assign_weights import assign_weights
from utils.make_matrix import gen_matrix
from utils.map_inputs import map_inputs
from utils.neuron_mapping import map_neurons


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    # Read the fields needed for mapping without expanding the large roiInfo field.
    neurons = pd.read_parquet(
        data_dir / "neurons.parquet",
        columns=[
            "bodyId", "type", "instance", "assignedOlHex1", "assignedOlHex2",
        ],
    )
    edges = pd.read_parquet(data_dir / "edges.parquet")
    mapped = map_neurons(neurons, edges)
    mapped["input_lookup"] = map_inputs(mapped["viz_neurons"])

    print(
        f"Mapped {len(mapped['neurons']):,} neurons, "
        f"{len(mapped['edges']):,} edges, and "
        f"{len(mapped['viz_neurons']):,} visual input neurons."
    )

    assign_weights(mapped['neurons'], mapped['edges'], np.random.default_rng())
    mapped["weight_matrix"] = gen_matrix(mapped["edges"], len(mapped["neurons"]))
    print(
        f"Built sparse weight matrix: {mapped['weight_matrix'].shape}, "
        f"{mapped['weight_matrix'].nnz:,} connections."
    )
    return mapped

if __name__ == "__main__":
    main()
