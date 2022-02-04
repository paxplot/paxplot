"""Core paxplot functions"""

from logging import warning
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import warnings


def scale_val(val, minimum, maximum):
    """
    Scale a value linearly between a minimum and maximum value

    Parameters
    ----------
    val : numeric
        Numeric value to be scaled
    minimum : numeric
        Minimum value to linearly scale between
    maximum : numeric
        Maximum value to lineraly scale between

    Returns
    -------
    val_scaled : numeric
        Scaled `val`
    """
    try:
        val_scaled = (val-minimum)/(maximum-minimum)
    except ZeroDivisionError:
        val_scaled = 0.5
    return val_scaled


def get_color_gradient(val, minimum, maximum, colormap):
    """
    Get color gradient values for the `val`

    Parameters
    ----------
    val : float
        value to get color for scaling
    minimum : float
        Minimum value
    maximum : float
        Minimum value for scaling
    colormap : str
        Matplotlib colormap to use for coloring

    Returns
    -------
    color: str
        string color code
    """
    color = mpl.colors.rgb2hex(
        cm.get_cmap(colormap)(scale_val(val, minimum, maximum))
    )
    return color


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
            ax.tick_params(axis='x', length=0.0, pad=10)

        # Adjust ticks on last axis
        self.axes[-1].yaxis.tick_right()

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

    def plot(self, data):
        """
        Plot the supplied data

        Parameters
        ----------
        data : array-like
            Data to be plotted
        """
        # Check n_axes
        if len(data[0]) < len(self.axes):
            warnings.warn(
                'Supplied data has fewer columns than figure. Figure created '
                'with empty column(s)',
                Warning
            )
        elif len(data[0]) > len(self.axes):
            raise ValueError(
                'Supplied data has fewer columns than figure. Please recreate '
                'paxfigure with appropriate n_axes'
            )

        # Convert to Numpy
        data = np.array(data)
        self.__setattr__('line_data', data)

        # Get data stats
        try:
            data_mins = data.min(axis=0)
            data_maxs = data.max(axis=0)
        except np.core._exceptions._UFuncNoLoopError:
            raise TypeError(
                'Non-plottable data has been supplied to argument `data`. '
                'Often this is caused by supplying non-numeric entries in '
                '`data`'
            )
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

            # Set attribute data
            self.axes[col_idx].__setattr__(
                'paxfig_lim',
                (data_mins[col_idx], data_maxs[col_idx])
            )

            # Defaults ticks
            self.set_even_ticks(
                    ax=self.axes[col_idx],
                    n_ticks=6,
                    minimum=data_mins[col_idx],
                    maximum=data_maxs[col_idx],
                    precision=2
                )

    def set_lim(self, ax_idx, bottom, top):
        """Set custom limits on axis

        Parameters
        ----------
        ax_idx : int
            Index of matplotlib axes
        bottom : numeric
            Lower limit
        top : numeric
            Upper limit
        """
        # Set default limits
        self.axes[ax_idx].set_ylim([0.0, 1.0])
        self.axes[ax_idx].__setattr__('paxfig_lim', (bottom, top))

        if ax_idx == 0:
            for i, line in enumerate(self.axes[ax_idx].lines):
                # Get y values
                y_data = self.line_data[i][[ax_idx, ax_idx+1]]

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
                ax=self.axes[ax_idx],
                n_ticks=6,
                minimum=bottom,
                maximum=top,
                precision=2
            )
        elif ax_idx < len(self.axes)-1:
            # Replace y first value
            for i, line in enumerate(self.axes[ax_idx].lines):
                y_data = self.line_data[i][[ax_idx, ax_idx+1]]
                y_0_scaled = scale_val(
                    val=y_data[0],
                    minimum=bottom,
                    maximum=top
                )
                line.set_ydata([y_0_scaled, line.get_ydata()[1]])

            # Replace the second y value
            for i, line in enumerate(self.axes[ax_idx-1].lines):
                y_data = self.line_data[i][[ax_idx-1, ax_idx]]
                y_1_scaled = scale_val(
                    val=y_data[1],
                    minimum=bottom,
                    maximum=top
                )
                line.set_ydata([line.get_ydata()[0], y_1_scaled])

            # Defaults ticks
            self.set_even_ticks(
                ax=self.axes[ax_idx],
                n_ticks=6,
                minimum=bottom,
                maximum=top,
                precision=2
            )

        elif ax_idx == len(self.axes)-1:
            # Work with second to last axis
            ax = self.axes[-2]
            ax_idx = len(self.axes)-2

            # Set the end of the line
            for i, line in enumerate(ax.lines):
                # Get y values
                y_data = self.line_data[i][[ax_idx, ax_idx+1]]

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

    def set_ticks(self, ax_idx, ticks, labels=None):
        """Set the axis tick locations and optionally labels.

        Parameters
        ----------
        ax_idx : int
            Index of matplotlib axes
        ticks : list of floats
            List of tick locations.
        labels : list of str, optional
            List of tick labels. If not set, the labels show the data value.
        """
        # Retrieve matplotlib axes
        ax = self.axes[ax_idx]

        # Set the limits if needed (this preserves matplotlib's
        # mandatory expansion of the view limits)
        try:
            [float(tick.get_text()) for tick in ax.get_yticklabels()]
        except ValueError:
            pass
        else:
            # Expand limits
            ticks_with_limits = list(ax.paxfig_lim)+ticks
            self.set_lim(
                    ax_idx=ax_idx,
                    bottom=min(ticks_with_limits),
                    top=max(ticks_with_limits)
                )

            # Scale the ticks
            minimum = min(ticks_with_limits)
            maximum = max(ticks_with_limits)
            tick_scaled = [scale_val(i, minimum, maximum) for i in ticks]

        # Set the ticks
        ax.set_yticks(ticks=tick_scaled)
        ax.set_yticklabels(labels=ticks)
        if labels is not None:
            ax.set_yticklabels(labels=labels)

    def set_label(self, ax_idx, label):
        """Set the label for the axis

        Parameters
        ----------
        ax_idx : int
            Index of matplotlib axes
        label : str
            The label text
        """
        ax = self.axes[ax_idx]
        ax.set_xticks(ticks=[0.0])
        ax.set_xticklabels([label])

    def invert_axis(self, ax_idx):
        """Invert axis.

        Parameters
        ----------
        ax_idx : int
            Index of matplotlib axes
        """
        # Local vars
        ax = self.axes[ax_idx]

        if ax_idx == 0:
            for line in ax.lines:
                # Flip y value about 0.5
                y_0_scaled = 1.0 - line.get_ydata()[0]

                # Replace the second y value
                line.set_ydata([y_0_scaled, line.get_ydata()[1]])
        elif ax_idx < len(self.axes)-1:
            # Flip left value
            for line in ax.lines:
                y_0_scaled = 1.0 - line.get_ydata()[0]
                line.set_ydata([y_0_scaled, line.get_ydata()[1]])
            # Flip right value
            for line in self.axes[ax_idx-1].lines:
                y_1_scaled = 1.0 - line.get_ydata()[1]
                line.set_ydata([line.get_ydata()[0], y_1_scaled])
        elif ax_idx == len(self.axes)-1:
            for line in self.axes[-2].lines:
                # Flip y value about 0.5
                y_1_scaled = 1.0 - line.get_ydata()[1]

                # Replace the second y value
                line.set_ydata([line.get_ydata()[0], y_1_scaled])

        # Invert ticks
        ticks = ax.get_yticks()
        ticks_scaled = 1.0 - ticks
        labels = [i.get_text() for i in ax.get_yticklabels()]
        ax.set_yticks(ticks=ticks_scaled)
        ax.set_yticklabels(labels=labels)

    def add_legend(self, label):
        """Create a legend for a specified figure

        Parameters
        ----------
        label : list
            List of data labels
        """
        # Set line labels
        for ax in self.axes:
            for i, line in enumerate(ax.lines):
                line.set_label(label[i])

        # Create blank axis for legend
        n_axes = len(self.axes)
        width_ratios = self.axes[0].get_gridspec().get_width_ratios()
        new_n_axes = n_axes + 1
        new_width_ratios = width_ratios + [1.0]
        gs = self.add_gridspec(1, new_n_axes, width_ratios=new_width_ratios)
        ax_legend = self.add_subplot(gs[0, n_axes])

        # Create legend
        lines = self.axes[0].lines
        labels = [i.get_label() for i in lines]
        ax_legend.legend(lines, labels, loc='center right')

        # Figure formatting
        for i in range(n_axes):
            self.axes[i].set_subplotspec(gs[0:1, i:i+1])
        ax_legend.set_axis_off()

    def add_colorbar(self, ax_idx, cmap='viridis', colorbar_kwargs={}):
        """Add colorbar to paxfigure

        Parameters
        ----------
        ax : int
            axes index
        data : array-like
            Data to be plotted
        cmap : str
            Matplotlib colormap to use for coloring
        colorbar_kwargs : dict
            Matplotlib colorbar keyword arguments
        """
        # Local vars
        n_lines = len(self.axes[0].lines)
        n_axes = len(self.axes)

        # Change line colors
        for i in range(n_lines):
            # Get value
            if ax_idx < len(self.axes)-1:
                scale_val = self.axes[ax_idx].lines[i].get_ydata()[0]
            else:
                scale_val = self.axes[ax_idx-1].lines[i].get_ydata()[1]
            # Get color
            color = get_color_gradient(scale_val, 0, 1, cmap)
            # Assign color to line
            for j in self.axes[:-1]:
                j.lines[i].set_color(color)

        # Create blank axis for colorbar
        width_ratios = self.axes[0].get_gridspec().get_width_ratios()
        new_n_axes = n_axes + 1
        new_width_ratios = width_ratios + [0.5]
        gs = self.add_gridspec(1, new_n_axes, width_ratios=new_width_ratios)
        ax_colorbar = self.add_subplot(gs[0, n_axes])

        # Create colorbar
        sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(
                vmin=self.axes[ax_idx].paxfig_lim[0],
                vmax=self.axes[ax_idx].paxfig_lim[1]
            ),
            cmap=cmap
        )
        self.colorbar(sm, orientation='vertical', **colorbar_kwargs)

        # Figure formatting
        for i in range(n_axes):
            self.axes[i].set_subplotspec(gs[0:1, i:i+1])
        ax_colorbar.set_axis_off()


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
    """
    # Check type of n_axes
    try:
        width_ratios = [1.0]*(n_axes-1)
    except TypeError:
        raise TypeError(
            f'n_axes should by of type int. You have supplied a type'
            f'{type(n_axes)}'
        )

    # Create figure
    width_ratios.append(0.0)  # Last axis small
    fig, _ = plt.subplots(
        1,
        n_axes,
        sharey=False,
        gridspec_kw={'width_ratios': width_ratios},
        FigureClass=PaxFigure,
    )
    fig.default_format()
    return fig
