"""Core paxplot functions"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
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


class PaxFigure(Figure):
    def __init__(self, *args, data=[], **kwargs):
        """
        Paxplot extension of Matplot Figure
        """
        super().__init__(*args, **kwargs)

    def default_format(self):
        """
        Set the default format of a Paxplot Figure
        """
        # Remove space between plots
        subplots_adjust_args = {
            'wspace': 0.0,
            'hspace': 0.0
        }
        self.subplots_adjust(**subplots_adjust_args)

        for ax in self.axes:
            # Remove axes frame
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Set limits
            ax.set_ylim([0, 1])
            ax.set_xlim([0, 1])

            # Set x ticks
            ax.set_xticks([0], [' '])

        # Last axis formatting
        self.axes[-1].yaxis.tick_right()  # TODO add test
        # self.axes[-1].tick_params(direction='in', pad=-22)  # TODO add test
        # self.axes[-1].set_box_aspect(0.1)
        # self.add_gridspec(width_ratios=[1.0, 1.0, 1.0, 0.1]) # TODO set here


class PaxAxes:
    def __init__(self, axes):
        self.axes = axes

    def plot(self, data):
        """
        Plot the supplied data

        Parameters
        ----------
        data : array-like
            Data to be plotted
        """
        # Convert to Numpy
        data = np.array(data)
        self.__setattr__('data', data)

        # Get data stats
        data_mins = data.min(axis=0)
        data_maxs = data.max(axis=0)
        n_rows = data.shape[0]
        n_cols = data.shape[1]

        for col_idx in range(n_cols):
            # Plot each line
            for row_idx in range(n_rows):
                if col_idx < n_cols - 1:  # Ignore last axis
                    # Scale the data
                    y_0_scaled = scale_val(
                        val=data[row_idx, col_idx],
                        minimum=data_mins[col_idx],
                        maximum=data_maxs[col_idx]
                    )
                    y_1_scaled = scale_val(
                        val=data[row_idx, col_idx + 1],
                        minimum=data_mins[col_idx + 1],
                        maximum=data_maxs[col_idx + 1]
                    )

                    # Plot the data
                    x = [0, 1]  # Assume each axes has a length between 0 and 1
                    y = [y_0_scaled, y_1_scaled]
                    self.axes[col_idx].plot(x, y)

            # Defaults ticks
            self.set_even_ticks(
                    ax=self.axes[col_idx],
                    n_ticks=6,
                    minimum=data_mins[col_idx],
                    maximum=data_maxs[col_idx],
                    precision=2
                )

    def set_even_ticks(self, ax, n_ticks, minimum, maximum, precision):
        """Set evenly spaced axis ticks between minimum and maximum value

        Parameters
        ----------
        ax : AxesSubplot
            Matplotlib axes
        n_ticks : int
            Number of ticks
        minimum : numeric
            minimum value for ticks
        maximum : numeric
            maximum value for ticks
        precision : int
            number of decimal points for tick labels
        """
        ticks = np.linspace(0, 1, num=n_ticks + 1)
        tick_labels = np.linspace(
            minimum,
            maximum,
            num=n_ticks + 1
        )
        tick_labels = tick_labels.round(precision)
        ax.set_yticks(ticks=ticks, labels=tick_labels)

    def set_ylim(self, ax, bottom, top):
        """Set custom y limits on axis

        Parameters
        ----------
        ax : AxesSubplot
            Matplotlib axes
        bottom : numeric
            Lower y limit
        top : numeric
            Upper y limit
        """
        # Get axis index
        ax_idx = np.where(self.axes == ax)[0][0]

        # For non-last axis
        if ax_idx < len(self.axes)-1:
            for i, line in enumerate(ax.lines):
                # Get y values
                y_data = self.data[i][[ax_idx, ax_idx+1]]

                # Scale the first y value
                y_0_scaled = scale_val(
                    val=y_data[0],
                    minimum=bottom,
                    maximum=top
                )

                # Replace y first value (keep the existing second)
                line.set_ydata([y_0_scaled, line.get_ydata()[1]])

            # Defaults ticks
            self.set_even_ticks(
                ax=ax,
                n_ticks=6,
                minimum=bottom,
                maximum=top,
                precision=2
            )

        # For last axis, set lines in previous axis
        elif ax_idx == len(self.axes)-1:
            # Work with second to last axis
            ax = self.axes[-2]
            ax_idx = len(self.axes)-2

            # Set the end of the line
            for i, line in enumerate(ax.lines):
                # Get y values
                y_data = self.data[i][[ax_idx, ax_idx+1]]

                # Scale the second y value
                y_1_scaled = scale_val(
                    val=y_data[1],
                    minimum=bottom,
                    maximum=top
                )

                # Replace the second y value
                line.set_ydata([line.get_ydata()[0], y_1_scaled])

            # Defaults ticks
            self.set_even_ticks(
                ax=self.axes[-1],
                n_ticks=6,
                minimum=bottom,
                maximum=top,
                precision=2
            )


def pax_parallel(n_axes):
    """
    Wrapper for paxplot analagous to the matplotlib.pyplot.subplots function

    Parameters
    ----------
    n_axes : int
        Number of axes to create

    Returns
    -------
    fig : PaxFigure
        Paxplot figure class
    axes : PaxAxes
        Paxplot axes class
    """
    width_ratios = [1.0]*(n_axes-1)
    width_ratios.append(0.0)  # Last axis small
    fig, axes = plt.subplots(
        1,
        n_axes,
        sharey=False,
        gridspec_kw={'width_ratios': width_ratios},
        FigureClass=PaxFigure,
    )
    fig.default_format()
    axes = PaxAxes(axes)
    return fig, axes
