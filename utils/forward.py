import numpy as np

from utils.map_inputs import image_to_activations


def forward_image(image, weight_matrix, input_lookup, steps=5):
    num_neurons = weight_matrix.shape[0]

    inputs = image_to_activations(image, input_lookup, num_neurons)
    activations = inputs.copy()
    input_indices = input_lookup.index.to_numpy()
    stats = []

    for step in range(1, steps + 1):
        activations = np.maximum(weight_matrix @ activations, 0)
        activations[input_indices] = inputs[input_indices]
        stats.append({
            "step": step,
            "mean": float(activations.mean()),
            "max": float(activations.max()),
            "active_neurons": int(np.count_nonzero(activations)),
        })

    return activations, stats
