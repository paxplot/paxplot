"""Core parapy functions"""

import matplotlib.pyplot as plt
import numpy as np

def parallel(
        data,
        cols
):
    """
    Create static parallel plot in Matplotlib.

    Shouts to Been Alex Keen for giving example of underlying logic on their
    website: https://benalexkeen.com/parallel-coordinates-in-matplotlib/

    :param data: list
        List of dictionaries containing the contents of data to be plotted.
        This has the form:
        [
            {col1: val1, col2: val1 ...},
            {col1: val2, col2: val2 ...}
            ...
        ]
    :param cols: list
        Columns to be plotted

    :return:
    """

    # Input error checking

    # Setting automatic values

    # Scale data

    # Create plot
    fig, axes = plt.subplots(1, len(cols) - 1, sharey=False)
    if len(cols) == 2:
        axes = [axes]

    # Plot each column pair at a time (axes)
    for ax_idx, ax in enumerate(axes):
        # Plot each line
        for row in data:
            y = [row[cols[ax_idx]], row[cols[ax_idx+1]]]
            x = [0, 1]
            ax.plot(x, y)

    # Format plots

    # Format ticks


    return fig
