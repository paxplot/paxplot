"""Core paxplot functions"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import warnings
import functools


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

    _safe_inherited_functions = [
        'savefig',
        'set_size_inches',
        'draw',
        'show'
    ]

    def __init__(self, *args, data=[], **kwargs):
        """
        Paxplot extension of Matplot Figure
        """
        # Setup
        super().__init__(*args, **kwargs)
        self._show_unsafe_warning = True

        # Paxplot attributes
        self._pax_data = []
        self._pax_data_scale = []
        self._pax_lims = []
        self._pax_ticks = []
        self._pax_ticks_scale = []
        self._pax_ticks_labels = []

    def default_format(self):
        """
        Set the default format of a Paxplot Figure
        """
        # Set attributes
        def_vals = [[0, 1]]*len(self.axes)
        self._pax_lims = def_vals.copy()
        self._pax_ticks = def_vals.copy()
        self._pax_ticks_scale = def_vals.copy()
        self._pax_ticks_labels = def_vals.copy()

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

            # Set y ticks
            ax.set_yticks([0, 1])

        # Adjust ticks on last axis
        self.axes[-1].yaxis.tick_right()

    def set_even_ticks(
        self,
        ax_idx: int,
        n_ticks=6,
        minimum=None,
        maximum=None,
        precision=2
    ):
        """
        Set evenly spaced axis ticks between minimum and maximum value. If
        no minimum and maximum values are specified, the limits of the
        underlying plotted data are assumed.

        Parameters
        ----------
        ax_idx : int
            Index of matplotlib axes
        n_ticks : int
            Number of ticks
        minimum : numeric
            minimum value for ticks
        maximum : numeric
            maximum value for ticks
        precision : int
            number of decimal points for tick labels
        """
        # Set automatic min and maximum
        if minimum is None and maximum is None:
            minimum = self.line_data[:, ax_idx].min()
            maximum = self.line_data[:, ax_idx].max()

        # Minimum/maximum check
        if minimum > maximum:
            raise ValueError(
                f'Value for `minimum` cannot be greater than `maximum`'
            )

        # Retrieve matplotlib axes
        try:
            ax = self.axes[ax_idx]
        except IndexError:
            raise IndexError(
                f'You are trying to set the limits of axis with index '
                f'{ax_idx}. However, axis index only goes up to '
                f'{len(self.axes)-1}.'
            )
        except TypeError:
            raise TypeError(
                f'Type of `ax_idx` must be integer not {type(ax_idx)}'
            )

        # Setting ticks
        try:
            ticks = np.linspace(0, 1, num=n_ticks + 1)
        except TypeError:
            raise TypeError(
                f'Type of `n_ticks` must be integer not {type(n_ticks)}'
            )
        tick_labels = np.linspace(
            minimum,
            maximum,
            num=n_ticks + 1
        )
        tick_labels = tick_labels.round(precision)
        ax.set_yticks(ticks=ticks, labels=tick_labels)

    def plot(self, data: list):
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
        data_input = np.array(data)

        # # Pull attributes
        # data = self._pax_data
        # data_scale = self._pax_data_scale
        # lims = self._pax_lims
        # ticks = self._pax_ticks
        # ticks_scales = self._pax_ticks_scale
        # ticks_labels = self._pax_ticks_labels

        # Set paxplot data attributes
        if not self._pax_data:
            self._pax_data = data_input
        else:
            self._pax_data.append(data_input)
        self._pax_data_scale = self._pax_data.copy()

        # Get data stats
        try:
            data_mins = self._pax_data.min(axis=0)
            data_maxs = self._pax_data.max(axis=0)
        except np.core._exceptions._UFuncNoLoopError:
            raise TypeError(
                'Non-plottable data has been supplied to argument `data`. '
                'Often this is caused by supplying non-numeric entries in '
                '`data`'
            )

        # Checking for singleton case
        for i in range(len(data_maxs)):
            if data_mins[i] == data_maxs[i]:
                data_mins[i] = data_mins[i]-1.0
                data_maxs[i] = data_maxs[i]+1.0

        # Scale data
        for col_idx, col in enumerate(self._pax_data.T):
            col = (col - np.min(col)) / (np.max(col) - np.min(col))
            self._pax_data_scale[:, col_idx] = col

        # Plotting scaled data (assume each axes has a x axis limits between
        # 0 and 1)
        for ax_idx, ax in enumerate(self.axes[:-1]):
            ax.plot(self._pax_data_scale[:, ax_idx:ax_idx+2].T)

        # Set limits
        self._pax_lims = list(map(list, zip(data_mins, data_maxs)))
        for ax_idx in range(len(self.axes)):
            self.set_lim(
                ax_idx=ax_idx,
                bottom=self._pax_lims[ax_idx][0],
                top=self._pax_lims[ax_idx][1]
            )

        # Generate uniform ticks
        

        # Set ticks


        

    def set_lim(self, ax_idx: int, bottom: float, top: float):
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
        # Check bottom top values
        try:
            if bottom > top:
                raise ValueError(
                    'Value for `bottom` cannot be greater than `top`. To '
                    'invert axis use the `invert_axis` function.'
                )
        except TypeError:
            raise TypeError(
                f'Both `bottom` and `top` must be numeric values. Currently '
                f'`bottom` is of type {type(bottom)} and `top` is of type'
                f'{type(top)}'
            )

        # Set default limits
        try:
            self.axes[ax_idx].set_ylim([0.0, 1.0])
        except IndexError:
            raise IndexError(
                f'You are trying to set the limits of axis with index '
                f'{ax_idx}. However, axis index only goes up to '
                f'{len(self.axes)-1}.'
            )
        except TypeError:
            raise TypeError(
                f'Type of `ax_idx` must be integer not {type(ax_idx)}'
            )

        # Scale data
        col = self._pax_data[:, ax_idx]
        col = (col - bottom) / (top - bottom)
        self._pax_data_scale[:, ax_idx] = col

        # Update plot of scaled data
        if ax_idx == 0:  # First axis
            for i, line in enumerate(self.axes[ax_idx].lines):
                # Replace left y value
                y_l_scaled = self._pax_data_scale[i, ax_idx]
                line.set_ydata(
                    [y_l_scaled, line.get_ydata()[1]]
                )
        elif ax_idx == self._pax_data.shape[1]-1:  # Last axis
            for i, line in enumerate(self.axes[ax_idx-1].lines):
                # Replace right y value
                y_r_scaled = self._pax_data_scale[i, ax_idx]
                line.set_ydata([line.get_ydata()[0], y_r_scaled])
        else:  # Middle axes
            for i, line in enumerate(self.axes[ax_idx].lines):
                # Replace left y value
                y_l_scaled = self._pax_data_scale[i, ax_idx]
                line.set_ydata(
                    [y_l_scaled, line.get_ydata()[1]]
                )
            for i, line in enumerate(self.axes[ax_idx-1].lines):
                # Replace right y value
                y_r_scaled = self._pax_data_scale[i, ax_idx]
                line.set_ydata([line.get_ydata()[0], y_r_scaled])

        # Scale ticks
        ticks = self._pax_ticks[ax_idx]
        ticks_scale = (ticks - bottom) / (top - bottom)
        self._pax_ticks_scale[ax_idx] = ticks_scale

        # Update ticks
        self.axes[ax_idx].set_yticks(ticks=ticks_scale)
        self.axes[ax_idx].set_yticklabels(
            labels=self._pax_ticks_labels[ax_idx]
        )

    def set_ticks(self, ax_idx: int, ticks: list, labels=None):
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
        # Tick tests ('ask permission' mindset as nested try/except gets nasty)
        try:
            ticks+[1]
        except TypeError:
            raise TypeError(
                f'`ticks` must be array-like not type {type(ticks)}'
            )
        try:
            min(ticks)
        except TypeError:
            raise TypeError(
                f'All entries in `ticks` must be numeric. To set string ticks,'
                f' use the `labels` argument'
            )

        # Checking if data is plotted
        try:
            self.line_data
        except AttributeError:
            raise AttributeError(
                'Paxplot does not support set_ticks if no data has been'
                'plotted'
            )

        # Retrieve matplotlib axes
        try:
            ax = self.axes[ax_idx]
        except IndexError:
            raise IndexError(
                f'You are trying to set the limits of axis with index '
                f'{ax_idx}. However, axis index only goes up to '
                f'{len(self.axes)-1}.'
            )
        except TypeError:
            raise TypeError(
                f'Type of `ax_idx` must be integer not {type(ax_idx)}'
            )

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
            try:
                ax.set_yticklabels(labels=labels)
            except ValueError:
                raise ValueError(
                    f'Length of `labels` must be same as length of `ticks`'
                )

    def set_label(self, ax_idx: int, label: str):
        """Set the label for the axis

        Parameters
        ----------
        ax_idx : int
            Index of matplotlib axes
        label : str
            The label text
        """
        try:
            ax = self.axes[ax_idx]
        except IndexError:
            raise IndexError(
                f'You are trying to set the limits of axis with index '
                f'{ax_idx}. However, axis index only goes up to '
                f'{len(self.axes)-1}.'
            )
        except TypeError:
            raise TypeError(
                f'Type of `ax_idx` must be integer not {type(ax_idx)}'
            )

        ax.set_xticks(ticks=[0.0])
        ax.set_xticklabels([label])

    def set_labels(self, labels: list):
        """
        Set labels for all axes. A wrapper for set_label

        Parameters
        ----------
        labels : list
            Labels for each axis. Must be same length as number of axes.
        """
        # Checking length
        if len(self.axes) != len(labels):
            raise IndexError(
                'Length of `labels` must equal number of axes'
            )

        # Set labels
        for i, label in enumerate(labels):
            self.set_label(i, label)

    def invert_axis(self, ax_idx: int):
        """Invert axis

        Parameters
        ----------
        ax_idx : int
            Index of matplotlib axes
        """
        # Local vars
        try:
            ax = self.axes[ax_idx]
        except IndexError:
            raise IndexError(
                f'You are trying to set the limits of axis with index '
                f'{ax_idx}. However, axis index only goes up to '
                f'{len(self.axes)-1}.'
            )
        except TypeError:
            raise TypeError(
                f'Type of `ax_idx` must be integer not {type(ax_idx)}'
            )

        # Checking if data is plotted
        try:
            self.line_data
        except AttributeError:
            raise AttributeError(
                'Paxplot does not support invert_axis if no data has been'
                'plotted'
            )

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

    def add_legend(self, labels: list):
        """Create a legend for a specified figure

        Parameters
        ----------
        labels : list
            List of data labels
        """
        # Check if too many labels supplied
        if len(labels) > len(self.axes[0].lines):
            warnings.warn(
                'More labels supplied than data. Some labels are unused.',
                Warning
            )

        # Set line labels
        try:
            for ax in self.axes:
                for i, line in enumerate(ax.lines):
                    line.set_label(labels[i])
        except IndexError:
            raise IndexError(
                f'Incorrect number of labels specified. You have supplied '
                f'{len(labels)} labels, but {len(ax.lines)} were expected'
            )

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

    def add_colorbar(self, ax_idx: int, cmap='viridis', colorbar_kwargs={}):
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

        # Testing
        try:
            self.axes[ax_idx]
        except IndexError:
            raise IndexError(
                f'You are trying to set the limits of axis with index '
                f'{ax_idx}. However, axis index only goes up to '
                f'{len(self.axes)-1}.'
            )
        except TypeError:
            raise TypeError(
                f'Type of `ax_idx` must be integer not {type(ax_idx)}'
            )

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


def add_unsafe_warning(func, fig):
    """
    Generate warning if not supported by Paxplot
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if fig._show_unsafe_warning:
            warnings.warn(
                f'The function you have called ({func.__name__}) is not '
                'officially supported by Paxplot, but it may still work. '
                'Report issues to '
                'https://github.com/kravitsjacob/paxplot/issues',
                Warning
            )
        return func(*args, **kwargs)
    return wrapper


def disable_unsafe_warnings(func, fig):
    """
    Temporarily disables safety warnings for the duration of the function
    execution.

    This allows a known safe function needs to make safe calls to otherwise
    unsafe functions without throwing a warning.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        original_flag_value = fig._show_unsafe_warning
        fig._show_unsafe_warning = False
        result = func(*args, **kwargs)
        fig._show_unsafe_warning = original_flag_value
        return result
    return wrapper


def pax_parallel(n_axes: int):
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

    pax_figure_functions = set(filter(
        lambda func_name: callable(getattr(PaxFigure, func_name)),
        vars(PaxFigure).keys()))

    unsafe_functions = set(filter(
        lambda func_name: (
            func_name not in PaxFigure._safe_inherited_functions
            and func_name not in pax_figure_functions),
        dir(Figure)))

    # Add unsafe function warnings
    for func_name in dir(PaxFigure):
        cond_1 = not func_name.startswith('__')
        cond_2 = not func_name.startswith('_')
        cond_3 = not func_name.startswith('get')
        cond_4 = callable(getattr(PaxFigure, func_name))
        if cond_1 and cond_2 and cond_3 and cond_4:
            func = getattr(fig, func_name)
            if func_name in unsafe_functions:
                func = add_unsafe_warning(func, fig)
            else:
                func = disable_unsafe_warnings(func, fig)
            setattr(fig, func_name, func)

    return fig
