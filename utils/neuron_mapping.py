import pandas as pd
import numpy as np

def map_neurons(neurons, edges):
    # Network indices match positions in the activation vector.
    neurons.reset_index(drop=True, inplace=True)
    id_to_index = pd.Series(np.arange(len(neurons)), index=neurons["bodyId"])

    edges["source"] = edges["bodyId_pre"].map(id_to_index)
    edges["target"] = edges["bodyId_post"].map(id_to_index)

    #from the paper, the x coordinate is the difference between the two assigned hexes
    #the y coordinate is the sum of the two assigned hexes divided by sqrt(3)
    viz_categories = ["L1_R", "L2_R", "L3_R"]
    viz_neurons = neurons.loc[
        neurons["instance"].isin(viz_categories),
        ["bodyId", "instance", "assignedOlHex1", "assignedOlHex2"],
    ]

    viz_neurons2d_x = viz_neurons["assignedOlHex2"] - viz_neurons["assignedOlHex1"]
    viz_neurons2d_y = (
        viz_neurons["assignedOlHex1"] + viz_neurons["assignedOlHex2"]
    ) / np.sqrt(3)

    neurons["fan_in"] = edges.groupby("target").size().reindex(neurons.index, fill_value=0)

    return {
        "neurons": neurons,
        "edges": edges,
        "viz_neurons2d_x": viz_neurons2d_x,
        "viz_neurons2d_y": viz_neurons2d_y,
        "identity_map": id_to_index,
    }
