import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.axes import Axes
from .funcs import *


"Colossogram Plot Function"
def plot_colossogram(xis_s, f, coherences, title=None, max_khz=None, cmap='magma'):
    # make meshgrid
    xx, yy = np.meshgrid(xis_s * 1000, f / 1000) # Note we convert xis to ms and f to kHz
    
    # Handle transpose if necessary
    if xx.shape[0] != coherences.shape[0]: 
        coherences = coherences.T
        
    # plot the heatmap
    vmin = 0
    vmax = 1
    heatmap = plt.pcolormesh(xx, yy, coherences, vmin=vmin, vmax=vmax, cmap=cmap, shading='nearest')

    # get and set label for cbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Vector Strength")
    if max_khz is not None:
        plt.ylim(0, max_khz)

    # set axes labels and titles
    plt.xlabel(rf"$\xi$ [ms]")
    plt.ylabel("Frequency [kHz]")
    if title is None:
        title = rf"Colossogram"
    plt.title(title)


