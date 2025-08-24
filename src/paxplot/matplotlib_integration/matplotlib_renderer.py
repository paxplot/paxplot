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
        self._axes: list = []  # Will contain secondary axes
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
        Internal method to create figure and main axis.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        **kwargs
            Additional arguments passed to plt.figure.
        """
        # Create figure with appropriate number of columns
        n_columns = self.plot_model.num_columns
        if n_columns == 0:
            raise ValueError("Cannot initialize figure with 0 columns")
            
        # Create figure
        self._figure = plt.figure(figsize=figsize, **kwargs)
        
        # Create the main plotting axis that spans the entire plot area
        self._main_axis = self._figure.add_axes((0.1, 0.1, 0.8, 0.8))
        
        # Set up the main axis coordinate system
        if n_columns == 1:
            self._main_axis.set_xlim((-0.5, 0.5))
        else:
            self._main_axis.set_xlim((0, n_columns - 1))
        
        # Create secondary axes positioned at each x-coordinate
        self._axes = []
        for i in range(n_columns):
            if n_columns == 1:
                x_coord = 0.0
            else:
                x_coord = i
            
            # Use secondary_yaxis with explicit positioning at x-coordinate
            # For data coordinates, we need to convert x_coord to the right position
            if n_columns == 1:
                # Single column case
                secondary_ax = self._main_axis.secondary_yaxis('right')
            else:
                # Multiple columns: position each at its x-coordinate
                # Convert data coordinate to axis coordinate (0-1 range)
                axis_pos = x_coord / (n_columns - 1) if n_columns > 1 else 0.5
                secondary_ax = self._main_axis.secondary_yaxis(axis_pos)
            
            self._axes.append(secondary_ax)
        
    def _setup_axes_formatting(self) -> None:
        """Apply default formatting to all axes."""
        # Format each secondary axis
        for i, ax in enumerate(self._axes):
            # Hide most spines, keep only the left spine for the vertical line
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Make the left spine visible and styled as the vertical line
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(1)
            ax.spines['left'].set_alpha(0.8)
            
            # Hide x-axis ticks
            ax.set_xticks([])
            
            # Set the same y-limits as the main axis
            ax.set_ylim(self._main_axis.get_ylim())
        
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
        self._setup_axes_formatting()
        
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
        """Plot continuous lines on the main axis."""
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
            main_ax.set_xlim((-0.5, 0.5))
        else:
            main_ax.set_xlim((0, self.plot_model.num_columns - 1))
        # Extend y limits slightly to prevent line cropping
        main_ax.set_ylim((-1.1, 1.1))
        
        # Hide the main axis ticks and spines
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
        """Update tick marks and labels for each axis."""
        for i, ax in enumerate(self._axes):
            try:
                # Try numeric ticks first
                tick_values = self.plot_model.get_numeric_ticks(i)
                tick_positions = self.plot_model.get_numeric_ticks_normalized(i)
            except TypeError:
                # Fall back to categorical ticks
                tick_labels = self.plot_model.get_categorical_ticks(i)
                tick_positions = self.plot_model.get_categorical_ticks_normalized(i)
                tick_values = tick_labels
            
            # Set the ticks and labels
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([str(val) for val in tick_values])
            
            # For the rightmost axis, put ticks on the right side
            if i == self.plot_model.num_columns - 1:
                ax.yaxis.tick_right()
                
    def get_figure(self) -> Figure:
        """Get the matplotlib figure."""
        self._ensure_initialized()
        assert self._figure is not None  # Type guard
        return self._figure
        
    def show(self) -> None:
        """Display the figure."""
        self._ensure_initialized()
        assert self._figure is not None  # Type guard
        self._figure.show()
        
    def save(self, filename: str, **kwargs) -> None:
        """Save the figure to a file."""
        self._ensure_initialized()
        assert self._figure is not None  # Type guard
        self._figure.savefig(filename, **kwargs)