from pathlib import Path

import pandas as pd

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

    print(
        f"Mapped {len(mapped['neurons']):,} neurons, "
        f"{len(mapped['edges']):,} edges, and "
        f"{len(mapped['viz_neurons2d_x']):,} visual input neurons."
    )
    return mapped

if __name__ == "__main__":
    main()
