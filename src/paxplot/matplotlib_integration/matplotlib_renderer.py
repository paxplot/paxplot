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
        self._line_objects: list = []
        
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
        Internal method to create figure and axes using positioned axes.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        **kwargs
            Additional arguments passed to plt.figure.
        """
        # Create figure with appropriate number of axes
        n_axes = self.plot_model.num_columns
        if n_axes == 0:
            raise ValueError("Cannot initialize figure with 0 columns")
            
        # Create figure
        self._figure = plt.figure(figsize=figsize, **kwargs)
        
        # Create the main plotting axis that spans the entire plot area
        self._main_axis = self._figure.add_axes((0.1, 0.1, 0.8, 0.8))
        
        # Calculate axis positions to align with x-coordinates
        # If we have n columns, the plot spans x = 0 to n-1
        # We need tick axes at each x position: 0, 1, 2, ..., n-1
        
        plot_left = 0.1
        plot_right = 0.9
        plot_width = plot_right - plot_left
        
        # Each axis should be positioned at its corresponding x-coordinate
        axis_width = 0.0  # Zero width axes so they don't block anything
        
        self._axes = []
        
        for i in range(n_axes):
            if n_axes == 1:
                # Special case: single column at center
                x_position = 0.5
            else:
                # Map column index to position in plot area
                x_position = i / (n_axes - 1)
            
            # Convert to figure coordinates
            left = plot_left + x_position * plot_width - axis_width / 2
            
            # [left, bottom, width, height]
            ax = self._figure.add_axes((left, 0.1, axis_width, 0.8))
            self._axes.append(ax)
            
        self._setup_axes_formatting()
        
    def _setup_axes_formatting(self) -> None:
        """Apply default formatting to all axes."""
        for i, ax in enumerate(self._axes):
            # Remove most spines but keep the left spine for vertical axes
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Keep left spine visible for vertical axis lines
            
            # Make the axis background transparent but keep the axis visible
            ax.set_facecolor('none')
            
            # Set limits - normalized data ranges from -1 to 1, with padding to prevent cropping
            ax.set_ylim([-1.1, 1.1])
            
            # Set x limits to match the main plot coordinate system
            if self.plot_model.num_columns == 1:
                ax.set_xlim([-0.5, 0.5])
            else:
                ax.set_xlim([0, self.plot_model.num_columns - 1])
            
            # Set x ticks
            ax.set_xticks([0], [' '])
            ax.tick_params(axis='x', length=0.0, pad=10)
            
            # Set y ticks for normalized data
            ax.set_yticks([-1, 1])
            
        # Move ticks on the last axis to the right side
        if self._axes:
            self._axes[-1].yaxis.tick_right()
        
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
            Additional arguments passed to plt.figure.
        """
        self._ensure_initialized(figsize, **kwargs)
        self._clear_existing_plots()
        self._plot_data()
        self._update_ticks()
        
    def _clear_existing_plots(self) -> None:
        """Clear all existing line plots."""
        # Clear the main figure area for continuous lines
        if self._figure and hasattr(self, '_main_axis'):
            # Remove existing lines from the main axis
            for artist in self._main_axis.get_children():
                if hasattr(artist, 'get_data'):
                    artist.remove()
        self._line_objects = []
        
    def _plot_data(self) -> None:
        """Plot continuous lines across the entire figure."""
        if self.plot_model.num_rows == 0:
            return
            
        # Get normalized data for plotting
        normalized_data = np.array([
            self.plot_model.get_column_values_normalized(i) 
            for i in range(self.plot_model.num_columns)
        ]).T
        
        # Use the main plotting axis that spans the entire plot area
        main_ax = self._main_axis
        
        # Set the plot area to span from 0 to n_columns-1
        if self.plot_model.num_columns == 1:
            main_ax.set_xlim([-0.5, 0.5])
        else:
            main_ax.set_xlim([0, self.plot_model.num_columns - 1])
        # Extend y limits slightly to prevent line cropping
        main_ax.set_ylim([-1.1, 1.1])
        
        # Hide the main axis (we only want the tick axes to be visible)
        main_ax.set_xticks([])
        main_ax.set_yticks([])
        for spine in main_ax.spines.values():
            spine.set_visible(False)
        
        # Plot continuous lines across the entire figure
        # This is the key advantage - one line per data row
        for row_idx in range(self.plot_model.num_rows):
            x_coords = list(range(self.plot_model.num_columns))
            y_coords = normalized_data[row_idx, :]
            line, = main_ax.plot(x_coords, y_coords, 'b-', alpha=0.7)
            self._line_objects.append(line)
            
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