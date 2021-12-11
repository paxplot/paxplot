"""Core parapy functions"""

import matplotlib.pyplot as plt


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

    :return: fig: matplotlib.figure.Figure
        Matplotlib figure

    """

    # Input error checking

    # Setting automatic values

    # Scale data

    # Create empty figures
    fig, axes = plt.subplots(1, len(cols) - 1, sharey=False)
    if len(cols) == 2:
        axes = [axes]

    # Plot each column pair at a time (axes)
    for ax_idx, ax in enumerate(axes):
        # Plot each line
        for row in data:
            y = [row[cols[ax_idx]], row[cols[ax_idx+1]]]
            x = [0, 1]  # Assume each axes has a length between 0 and 1
            ax.plot(x, y)
            ax.set_xlim(x)
        # X axis formatting
        ax.spines['top'].set_visible(False)  # Remove axes frame
        ax.spines['bottom'].set_visible(False)  # Remove axes frame
        ax.set_xticks([0], cols[ax_idx])  # Set label

    # Last axis formatting
    axes[-1].set_xticks(list(ax.get_xticks())+[1], cols[-2:])
    axes[-1].yaxis.set_ticks_position('both')
    axes[-1].tick_params(labelright=True)

    # Remove space between plots
    subplots_adjust_args = {
        'wspace': 0.0,
        'hspace': 0.0
    }
    fig.subplots_adjust(**subplots_adjust_args)

    # Format ticks
    return fig
