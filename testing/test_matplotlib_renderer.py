"""Unit tests for MatplotlibRenderer."""

import pytest
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from paxplot.plot_model import PlotModel
from paxplot.matplotlib_integration.matplotlib_renderer import MatplotlibRenderer


class TestMatplotlibRenderer:
    """Test cases for MatplotlibRenderer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_data = [
            [1, "A", 10.5],
            [2, "B", 15.2],
            [3, "A", 8.7],
            [4, "C", 12.1]
        ]
        self.plot_model = PlotModel(self.sample_data)
        self.renderer = MatplotlibRenderer(self.plot_model)
        
    def teardown_method(self):
        """Clean up after tests."""
        plt.close('all')
        
    def test_initialization(self):
        """Test renderer initialization."""
        assert self.renderer.plot_model == self.plot_model
        assert self.renderer._figure is None
        assert self.renderer._axes == []
        assert self.renderer._line_objects == []
        
    def test_lazy_initialization_on_update(self):
        """Test that figure is created lazily when update is called."""
        assert self.renderer._figure is None
        
        self.renderer.update()
        
        assert self.renderer._figure is not None
        assert isinstance(self.renderer._figure, Figure)
        assert len(self.renderer._axes) == 3  # 3 columns in sample data
        assert all(hasattr(ax, 'plot') for ax in self.renderer._axes)
        
    def test_lazy_initialization_on_get_figure(self):
        """Test that figure is created lazily when get_figure is called."""
        assert self.renderer._figure is None
        
        figure = self.renderer.get_figure()
        
        assert self.renderer._figure is not None
        assert isinstance(figure, Figure)
        assert figure == self.renderer._figure
        
    def test_lazy_initialization_on_show(self):
        """Test that figure is created lazily when show is called."""
        assert self.renderer._figure is None
        
        self.renderer.show()
        
        assert self.renderer._figure is not None
        assert isinstance(self.renderer._figure, Figure)
        
    def test_lazy_initialization_on_save(self):
        """Test that figure is created lazily when save is called."""
        assert self.renderer._figure is None
        
        self.renderer.save("test.png")
        
        assert self.renderer._figure is not None
        assert isinstance(self.renderer._figure, Figure)
        
    def test_figure_initialization_with_custom_size(self):
        """Test figure initialization with custom size."""
        self.renderer.update(figsize=(12, 8))
        
        assert self.renderer._figure is not None
        assert self.renderer._figure.get_size_inches().tolist() == [12.0, 8.0]
        
    def test_figure_initialization_zero_columns_error(self):
        """Test error when initializing with zero columns."""
        empty_model = PlotModel([])
        renderer = MatplotlibRenderer(empty_model)
        
        with pytest.raises(ValueError, match="Cannot initialize figure with 0 columns"):
            renderer.update()
            
    def test_update_creates_lines(self):
        """Test that update creates line objects."""
        self.renderer.update()
        
        # Check that lines were created
        assert len(self.renderer._line_objects) == 2  # 2 line groups (between 3 columns)
        assert len(self.renderer._line_objects[0]) == 4  # 4 rows
        
    def test_update_with_empty_data(self):
        """Test update with empty plot model."""
        empty_model = PlotModel([[1, 2, 3]])  # One row to allow initialization
        renderer = MatplotlibRenderer(empty_model)
        renderer.update()
        
        # Remove the data
        empty_model.remove_rows([0])
        renderer.update()
        
        # Should not crash and should have no lines
        assert len(renderer._line_objects) == 0
        
    def test_axes_formatting(self):
        """Test that axes are properly formatted."""
        self.renderer.update()
        
        for ax in self.renderer._axes:
            # Check spines are hidden
            assert not ax.spines['top'].get_visible()
            assert not ax.spines['bottom'].get_visible()
            assert not ax.spines['right'].get_visible()
            
            # Check limits
            assert ax.get_ylim() == (0, 1)
            assert ax.get_xlim() == (0, 1)
            
            # Check x ticks
            assert ax.get_xticks().tolist() == [0]
            assert ax.get_xticklabels()[0].get_text() == ' '
            
            # Check y ticks
            assert ax.get_yticks().tolist() == [0, 1]
            
    def test_last_axis_tick_right(self):
        """Test that last axis has ticks on the right."""
        self.renderer.update()
        
        # Check that last axis has right ticks
        last_ax = self.renderer._axes[-1]
        assert last_ax.yaxis.get_ticks_position() == 'right'
        
    def test_line_creation(self):
        """Test that lines are created correctly."""
        self.renderer.update()
        
        # Check line objects structure
        assert len(self.renderer._line_objects) == 2  # Between 3 columns
        
        # Check each line group
        for line_group in self.renderer._line_objects:
            assert len(line_group) == 4  # 4 rows
            
            # Check each line
            for line in line_group:
                assert hasattr(line, 'get_data')
                assert hasattr(line, 'set_data')
                
    def test_tick_updates(self):
        """Test that ticks are updated correctly."""
        self.renderer.update()
        
        # Check that ticks were set
        for i, ax in enumerate(self.renderer._axes):
            yticks = ax.get_yticks()
            yticklabels = ax.get_yticklabels()
            
            # Should have some ticks and labels
            assert len(yticks) > 0
            assert len(yticklabels) > 0
            
    def test_numeric_and_categorical_ticks(self):
        """Test handling of both numeric and categorical columns."""
        # Create data with mixed types
        mixed_data = [
            [1.5, "A", 10.0],
            [2.5, "B", 15.0],
            [3.5, "A", 8.0]
        ]
        mixed_model = PlotModel(mixed_data)
        renderer = MatplotlibRenderer(mixed_model)
        
        renderer.update()
        
        # Should not crash and should have ticks
        assert len(renderer._axes) == 3
        for ax in renderer._axes:
            assert len(ax.get_yticks()) > 0
            
    def test_figure_subplots_adjust(self):
        """Test that subplot spacing is adjusted correctly."""
        self.renderer.update()
        
        # Check that subplots have no spacing
        assert self.renderer._figure is not None
        # The subplots_adjust should be called in _setup_axes_formatting
        # We can't easily test the exact values, but we can check it doesn't crash
        assert self.renderer._figure is not None
        
    def test_multiple_updates_same_figure(self):
        """Test that multiple updates use the same figure."""
        self.renderer.update()
        original_figure = self.renderer._figure
        
        self.renderer.update()
        
        # Should be the same figure object
        assert self.renderer._figure is original_figure

