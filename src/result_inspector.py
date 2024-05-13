"""
This script is here to visualize the correct and predicted temperatures
from the neural networks.
"""

import argparse
from matplotlib import pyplot as plt
from ising import load_configurations_bin
import numpy as np
import os

def plot(model_results, data, labels):
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    
    ax[0].scatter(model_results[:, 0], model_results[:, 1], picker=1)
    ax[1].set_axis_off()
    marker = ax[0].scatter(model_results[0, 0], model_results[0, 1], color="red")

    def on_click(event):
        index = event.ind[0]
        temp_init, temp_pred = model_results[index]
        label_index = np.argwhere(np.isclose(temp_init, labels)).flatten()[0]
        ax[1].cla()
        ax[1].imshow(data[label_index], cmap="summer")
        ax[1].set_axis_off()
        ax[1].set_title(f"initial: {temp_init}, preicted: {temp_pred}")
        marker.set_offsets((temp_init, temp_pred))
        fig.canvas.draw_idle()

    fig.canvas.callbacks.connect("pick_event", on_click)

    plt.show()

def main(model_results, data, labels):
    for file in [model_results, data, labels]:
        assert os.path.isfile(file), f"File {file} does not exist"

    data = load_configurations_bin(data)
    labels = np.loadtxt(labels, delimiter=",")
    model_results = np.loadtxt(model_results, delimiter=",")

    plot(model_results, data, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize corret and predicted temperatures by NN model")
    parser.add_argument("model_results", help="The csv file containing the predictions of the NN model")
    parser.add_argument("data", help="The .ising file containing the data used for prediction")
    parser.add_argument("labels", help="The csv file containing the labels for the data used for prediction")
    parsed = parser.parse_args()
    main(parsed.model_results, parsed.data, parsed.labels)
