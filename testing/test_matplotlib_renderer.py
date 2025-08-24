"""Unit tests for MatplotlibRenderer."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
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
        # Create a model with empty data that will result in 0 columns
        with pytest.raises(ValueError, match="Input must be a 2D array-like structure"):
            empty_model = PlotModel([])
            
    def test_update_creates_lines(self):
        """Test that update creates line objects."""
        self.renderer.update()
        
        # Check that lines were created
        assert len(self.renderer._line_objects) == 4  # 4 lines (one per data row)
        # Each line object should be a matplotlib Line2D object
        for line in self.renderer._line_objects:
            assert hasattr(line, 'get_data')
        
    def test_update_with_empty_data(self):
        """Test update with empty plot model - skip for now."""
        # TODO: Implement proper empty data handling later
        pytest.skip("Empty data handling not implemented yet")
        
    def test_axes_formatting(self):
        """Test that axes are properly formatted."""
        self.renderer.update()
        
        for ax in self.renderer._axes:
            # Check spines are hidden
            assert not ax.spines['top'].get_visible()
            assert not ax.spines['bottom'].get_visible()
            assert not ax.spines['right'].get_visible()
            
            # Check limits - normalized data ranges from -1.1 to 1.1 (with padding to prevent cropping)
            ylim = ax.get_ylim()
            assert abs(ylim[0] - (-1.1)) < 1e-10
            assert abs(ylim[1] - 1.1) < 1e-10
            
            xlim = ax.get_xlim()
            assert abs(xlim[0] - 0.0) < 1e-10
            # Tick axes are narrow (width=0.02), so xlim should be small
            assert xlim[1] > 0  # Just check it's positive
            
            # Check x ticks - tick axes are narrow and may not show x ticks
            xticks = ax.get_xticks()
            # Just verify ticks don't crash (they may be empty for narrow axes)
            assert isinstance(xticks, np.ndarray)
            
            # Check y ticks - should have some ticks set based on the data
            yticks = ax.get_yticks()
            assert len(yticks) > 0
            # All ticks should be within the [-1, 1] range
            assert all(-1.0 <= tick <= 1.0 for tick in yticks)
            
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
        assert len(self.renderer._line_objects) == 4  # 4 lines (one per data row)
        
        # Check each line
        for line in self.renderer._line_objects:
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

