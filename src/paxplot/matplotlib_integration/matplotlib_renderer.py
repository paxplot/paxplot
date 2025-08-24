"""Matplotlib renderer for PlotModel visualization."""

from typing import Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

from paxplot.plot_model import PlotModel


class MatplotlibRenderer:
    """
    Renders a PlotModel using matplotlib.
    
    This renderer creates and maintains matplotlib figures and axes,
    handling the visual representation of the plot model data.
    """
    
    def __init__(self, plot_model: PlotModel):
        """
        Initialize the renderer with a plot model.
        
        Parameters
        ----------
        plot_model : PlotModel
            The plot model to render.
        """
        self.plot_model = plot_model
        self._figure: Optional[Figure] = None
        self._axes: list[Axes] = []
        self._line_objects: list[list] = []
        
    def _ensure_initialized(self, figsize: tuple = (10, 6), **kwargs) -> None:
        """
        Ensure figure is initialized, create if needed.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        **kwargs
            Additional arguments passed to plt.subplots.
        """
        if self._figure is None:
            self._initialize_figure(figsize, **kwargs)
            
    def _initialize_figure(self, figsize: tuple = (10, 6), **kwargs) -> None:
        """
        Internal method to create figure and axes.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        **kwargs
            Additional arguments passed to plt.subplots.
        """
        # Create figure with appropriate number of axes
        n_axes = self.plot_model.num_columns
        if n_axes == 0:
            raise ValueError("Cannot initialize figure with 0 columns")
            
        width_ratios = [1.0] * (n_axes - 1) + [0.0]  # Last axis small for ticks
        
        self._figure, axes = plt.subplots(
            1, n_axes, 
            figsize=figsize,
            gridspec_kw={'width_ratios': width_ratios},
            **kwargs
        )
        
        # Convert single axis to list for consistency
        if n_axes == 1:
            self._axes = [axes]
        else:
            self._axes = list(axes)
            
        self._setup_axes_formatting()
        
    def _setup_axes_formatting(self) -> None:
        """Apply default formatting to all axes."""
        for i, ax in enumerate(self._axes):
            # Remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set limits - normalized data ranges from -1 to 1
            ax.set_ylim([-1, 1])
            ax.set_xlim([0, 1])
            
            # Set x ticks
            ax.set_xticks([0], [' '])
            ax.tick_params(axis='x', length=0.0, pad=10)
            
            # Set y ticks for normalized data
            ax.set_yticks([-1, 1])
            
        # Adjust ticks on last axis
        if self._axes:
            self._axes[-1].yaxis.tick_right()
            
        # Remove space between plots
        self._figure.subplots_adjust(wspace=0.0, hspace=0.0)
        
    def update(self, figsize: tuple = (10, 6), **kwargs) -> None:
        """
        Update the visualization to reflect the current state of the plot model.
        
        This method redraws all data and updates ticks based on the current
        plot model state.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        **kwargs
            Additional arguments passed to plt.subplots.
        """
        self._ensure_initialized(figsize, **kwargs)
        self._clear_existing_plots()
        self._plot_data()
        self._update_ticks()
        
    def _clear_existing_plots(self) -> None:
        """Clear all existing line plots."""
        for ax in self._axes:
            ax.clear()
        self._setup_axes_formatting()
        self._line_objects = []
        
    def _plot_data(self) -> None:
        """Plot the current data from the plot model."""
        if self.plot_model.num_rows == 0:
            return
            
        # Get normalized data for plotting
        normalized_data = np.array([
            self.plot_model.get_column_values_normalized(i) 
            for i in range(self.plot_model.num_columns)
        ]).T
        
        # Plot lines between adjacent columns
        for i in range(self.plot_model.num_columns - 1):
            lines = []
            for row_idx in range(self.plot_model.num_rows):
                x_data = [i, i + 1]
                y_data = [normalized_data[row_idx, i], normalized_data[row_idx, i + 1]]
                line, = self._axes[i].plot(x_data, y_data, 'b-', alpha=0.7)
                lines.append(line)
            self._line_objects.append(lines)
            
    def _update_ticks(self) -> None:
        """Update tick marks and labels for all axes."""
        for col_idx in range(self.plot_model.num_columns):
            try:
                # Try numeric ticks first
                tick_values = self.plot_model.get_numeric_ticks(col_idx)
                tick_positions = self.plot_model.get_numeric_ticks_normalized(col_idx)
                self._axes[col_idx].set_yticks(tick_positions)
                self._axes[col_idx].set_yticklabels([str(v) for v in tick_values])
            except TypeError:
                # Fall back to categorical ticks
                tick_labels = self.plot_model.get_categorical_ticks(col_idx)
                tick_positions = self.plot_model.get_categorical_ticks_normalized(col_idx)
                self._axes[col_idx].set_yticks(tick_positions)
                self._axes[col_idx].set_yticklabels(tick_labels)
                
    def get_figure(self) -> Figure:
        """Get the matplotlib figure."""
        self._ensure_initialized()
        return self._figure
        
    def show(self) -> None:
        """Display the figure."""
        self._ensure_initialized()
        self._figure.show()
        
    def save(self, filename: str, **kwargs) -> None:
        """Save the figure to a file."""
        self._ensure_initialized()
        self._figure.savefig(filename, **kwargs)