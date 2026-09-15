import numpy as np

def map_inputs(viz_neurons):
    x_min = viz_neurons["x"].min()
    x_max = viz_neurons["x"].max()
    y_min = viz_neurons["y"].min()
    y_max = viz_neurons["y"].max()

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    span = max(x_max - x_min, y_max - y_min)
    scale = 31 / span if span > 0 else 1.0

    viz_neurons["pixel_x"] = (viz_neurons["x"] - x_mid) * scale + 15.5
    viz_neurons["pixel_y"] =  15.5 - (viz_neurons["y"] - y_mid) * scale

    viz_neurons["channel"] = viz_neurons["instance"].map({
        "L1_R": 0,
        "L2_R": 1,
        "L3_R": 2,
    })

    # Keep pixel centers inside the image, including floating-point roundoff.
    viz_neurons[["pixel_x", "pixel_y"]] = viz_neurons[["pixel_x", "pixel_y"]].clip(0, 31)

    # Four neighbors: (x0, y0), (x1, y0), (x0, y1), (x1, y1).
    viz_neurons["x0"] = np.floor(viz_neurons["pixel_x"]).astype(int)
    viz_neurons["y0"] = np.floor(viz_neurons["pixel_y"]).astype(int)
    viz_neurons["x1"] = (viz_neurons["x0"] + 1).clip(upper=31)
    viz_neurons["y1"] = (viz_neurons["y0"] + 1).clip(upper=31)

    dx = viz_neurons["pixel_x"] - viz_neurons["x0"]
    dy = viz_neurons["pixel_y"] - viz_neurons["y0"]
    viz_neurons["w00"] = (1 - dx) * (1 - dy)
    viz_neurons["w10"] = dx * (1 - dy)
    viz_neurons["w01"] = (1 - dx) * dy
    viz_neurons["w11"] = dx * dy

    return viz_neurons

def image_to_activations(image, lookup, num_neurons):
    image = image.astype(np.float32) / 255.0
    activations = np.zeros(num_neurons, dtype=np.float32)

    for neuron in lookup.itertuples():
        activations[neuron.Index] = (
            image[neuron.y0, neuron.x0, neuron.channel] * neuron.w00 +
            image[neuron.y0, neuron.x1, neuron.channel] * neuron.w10 +
            image[neuron.y1, neuron.x0, neuron.channel] * neuron.w01 +
            image[neuron.y1, neuron.x1, neuron.channel] * neuron.w11
        )

    return activations
