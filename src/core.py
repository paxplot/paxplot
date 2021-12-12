"""Core parapy functions"""

import matplotlib.pyplot as plt
import numpy as np


def scale_val(val, minimum, maximum):
    """
    Scale a value linearly between a minimum and maximum value

    :param val: numeric
        Numeric value to be scaled
    :param minimum: numeric
        Minimum value to linearly scale between
    :param maximum: numeric
        Maximum value to lineraly scale between
    :return: val_scaled:numeric
        Scale `val`
    """
    try:
        val_scaled = (val-minimum)/(maximum-minimum)
    except ZeroDivisionError:
        val_scaled = 0.5
    return val_scaled


def get_data_lims(data, cols):
    """
    Get minimum and maximum value for each column

    :param data: list
        List of dictionaries containing the contents of data from `parallel`
    :param cols: list
        Columns to be plotted from `parallel`
    :return: cols_lims: dict
        Dictionary of column limits corresponding to columns in `cols` in form:
        {
            col1: [lower, upper],
            col2: [lower, upper],
            ...
        }
    """
    cols_lims = {}
    for col in cols:
        col_data = [row[col] for row in data]
        cols_lims[col] = [min(col_data), max(col_data)]

    return cols_lims


def parallel(
        data,
        cols,
        custom_lims=None
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
    :param custom_lims: dict
        Dictionary of custom column limits corresponding to columns in `cols`.
        Must be of the form:
        {
            col1: [lower, upper],
            col2: [lower, upper],
            ...
        }


    :return: fig: matplotlib.figure.Figure
        Matplotlib figure

    """

    # Input error checking

    # Setting automatic column limits
    if custom_lims is not None:
        cols_lims = custom_lims
    else:
        cols_lims = get_data_lims(data, cols)

    # Create empty figures
    fig, axes = plt.subplots(1, len(cols) - 1, sharey=False)
    if len(cols) == 2:
        axes = [axes]

    # Plot each column pair at a time (axes)
    for ax_idx, ax in enumerate(axes):
        # Plot each line
        for row in data:
            x = [0, 1]  # Assume each axes has a length between 0 and 1
            # Scale the data
            y_0_scaled = scale_val(
                val=row[cols[ax_idx]],
                minimum=cols_lims[cols[ax_idx]][0],
                maximum=cols_lims[cols[ax_idx]][1]
            )
            y_1_scaled = scale_val(
                val=row[cols[ax_idx + 1]],
                minimum=cols_lims[cols[ax_idx + 1]][0],
                maximum=cols_lims[cols[ax_idx + 1]][1]
            )
            y = [y_0_scaled, y_1_scaled]

            # Plot the data
            ax.plot(x, y)
            ax.set_xlim(x)
        # X axis formatting
        ax.spines['top'].set_visible(False)  # Remove axes frame
        ax.spines['bottom'].set_visible(False)  # Remove axes frame
        ax.set_xticks([0], cols[ax_idx])  # Set label

        # Y axis formatting
        # Set limits
        ax.set_ylim([0, 1])

        # Change ticks to reflect scaled data
        n_ticks = 10  # TODO make this a parameter in the future
        precision = 2  # TODO make this a parameter in the future
        tick_labels = np.linspace(
            cols_lims[cols[ax_idx]][0],
            cols_lims[cols[ax_idx]][1],
            num=n_ticks+1
        )
        tick_labels = tick_labels.round(precision)
        ticks = np.linspace(0, 1, num=n_ticks+1)

        ax.set_yticks(ticks=ticks, labels=tick_labels)

    # Last axes formatting
    last_ax = plt.twinx(axes[-1])
    last_ax.spines['top'].set_visible(False)  # Remove axes frame
    last_ax.spines['bottom'].set_visible(False)  # Remove axes frame
    n_ticks = 10  # TODO make this a parameter in the future
    precision = 2  # TODO make this a parameter in the future
    tick_labels = np.linspace(
        cols_lims[cols[-1]][0],
        cols_lims[cols[-1]][1],
        num=n_ticks + 1
    )
    tick_labels = tick_labels.round(precision)
    ticks = np.linspace(0, 1, num=n_ticks + 1)
    last_ax.set_yticks(ticks=ticks, labels=tick_labels)
    last_ax.set_xticks([0, 1], cols[-2:])  # Set label

    # Remove space between plots
    subplots_adjust_args = {
        'wspace': 0.0,
        'hspace': 0.0
    }
    fig.subplots_adjust(**subplots_adjust_args)

    # Format ticks
    return fig
