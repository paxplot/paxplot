"""Core parapy functions"""

import matplotlib.pyplot as plt


def scale_data(val, minimum, maximum):
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
    val_scaled = (val-minimum)/(maximum-minimum)
    return val_scaled


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

    # Setting automatic values

    # Create empty figures
    fig, axes = plt.subplots(1, len(cols) - 1, sharey=False)
    if len(cols) == 2:
        axes = [axes]

    # Plot each column pair at a time (axes)
    for ax_idx, ax in enumerate(axes):
        # Plot each line
        for row in data:
            x = [0, 1]  # Assume each axes has a length between 0 and 1
            if custom_lims is not None:
                # Custom limits scaling
                y_0_scaled = scale_data(
                    val=row[cols[ax_idx]],
                    minimum=custom_lims[cols[ax_idx]][0],
                    maximum=custom_lims[cols[ax_idx]][1]
                )
                y_1_scaled = scale_data(
                    val=row[cols[ax_idx + 1]],
                    minimum=custom_lims[cols[ax_idx + 1]][0],
                    maximum=custom_lims[cols[ax_idx + 1]][1]
                )
                y = [y_0_scaled, y_1_scaled]
            else:
                # If no scaling applied
                y = [row[cols[ax_idx]], row[cols[ax_idx + 1]]]
            # Plot the data
            ax.plot(x, y)
            ax.set_xlim(x)
        # X axis formatting
        ax.spines['top'].set_visible(False)  # Remove axes frame
        ax.spines['bottom'].set_visible(False)  # Remove axes frame
        ax.set_xticks([0], cols[ax_idx])  # Set label
        # Y axis formatting
        if custom_lims is not None:
            ax.set_ylim([0, 1])

    # Last axis formatting
    last_ax = axes[-1]
    last_ax.set_xticks(list(last_ax.get_xticks())+[1], cols[-2:])
    last_ax.yaxis.set_ticks_position('both')
    last_ax.tick_params(labelright=True)

    # Remove space between plots
    subplots_adjust_args = {
        'wspace': 0.0,
        'hspace': 0.0
    }
    fig.subplots_adjust(**subplots_adjust_args)

    # Format ticks
    return fig
