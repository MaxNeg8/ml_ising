import argparse
import os

import numpy as np

from ising import load_configurations_bin, plot_configuration

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

def main(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} does not exist")
    
    labels_file = filename.replace("_data_", "_labels_").replace(".ising", ".csv")
    show_labels = os.path.isfile(labels_file)

    configs = load_configurations_bin(filename)
    
    if show_labels:
        labels = np.loadtxt(labels_file, delimiter=",")
    
    if configs.ndim == 2:
        plot_configuration(configs)
        return

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.set_axis_off()

    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax_slider, "Configuration", 1, configs.shape[0], valinit=1, valstep=1)

    plot = ax.imshow(configs[0], cmap="summer", vmin=-1, vmax=1)
    if show_labels:
        ax.set_title(f"Temperature: {labels[0]}")

    def update(val):
        plot.set_data(configs[val - 1])
        ax.set_title(f"Temperature: {labels[val - 1]}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple tool to inspect ising data set")
    parser.add_argument("filename", help="The file to be inspected")
    filename = parser.parse_args().filename
    try:
        main(filename)
    except FileNotFoundError as err:
        print(err)
