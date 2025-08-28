# PaxPlot Architecture Design Document

## Overview

This document outlines the refactored architecture for PaxPlot, a parallel coordinate plotting library. The new design separates data management from normalization and rendering concerns, creating a more flexible and maintainable system.

## Core Principles

1. **Separation of Concerns**: Data structures are independent of normalization and rendering
2. **Pull-Based Updates**: Data views pull fresh data when needed
3. **View Pattern**: Normalization provided through lightweight data views
4. **Renderer Agnostic**: Core system works with any rendering backend

## Architecture Components

### Core Data Structures

#### `NumericalArray`
Stores numerical data as an array with basic operations.

```python
class NumericalArray:
    def __init__(self, values: Sequence[float])
    def append(self, value: float)
    def remove(self, index: int)
    @property
    def values(self) -> List[float]
    @property
    def custom_min(self) -> float
    @property
    def custom_max(self) -> float
```

#### `CategoricalArray`
Stores categorical (string) data as an array.

```python
class CategoricalArray:
    def __init__(self, values: Sequence[str])
    def append(self, value: str)
    def remove(self, index: int)
    @property
    def values(self) -> List[str]
    @property
    def unique_values(self) -> List[str]
    # This have a NumericalArray as a backend ent of just ints corresponding to the categorical values
```

#### `Matrix`
Manages collections of arrays, providing a unified interface.

```python
class Matrix:
    def __init__(self, data: Sequence[Sequence[Union[str, int, float]]])
    def get_column(self, index: int) -> Union[NumericalArray, CategoricalArray]
    def append_data(self, row: Sequence[Union[str, int, float]])
    def remove_data(self, index: int)
```

### Tick Management

#### `AxisTicks`
Base tick information containing positions and labels.

```python
class AxisTicks:
    def __init__(self, positions: List[float], labels: List[str])
    @property
    def positions(self) -> List[float]
    @property
    def labels(self) -> List[str]
```

#### `NumericAxisTicks`
Manages numeric tick generation and customization.

```python
class NumericAxisTicks:
    def __init__(self, array: NumericalArray)
    def generate_ticks(self, num_ticks: int = 5) -> AxisTicks
    def set_custom_ticks(self, tick_values: List[float]) -> AxisTicks
```

#### `CategoricalAxisTicks`
Manages categorical tick generation and customization.

```python
class CategoricalAxisTicks:
    def __init__(self, array: CategoricalArray)
    def generate_ticks(self) -> AxisTicks
    def set_custom_ticks(self, tick_labels: List[str]) -> AxisTicks
    # This will use the numeric array backend of the categorical array to set the ticks
```

#### `AxisTickManager`
Coordinates tick management across all columns.

```python
class AxisTickManager:
    def __init__(self, matrix: Matrix)
    def get_ticks(self, column_index: int) -> AxisTicks
    def set_custom_ticks(self, column_index: int, ticks: AxisTicks)
```

### Plot Model

#### `PlotModel`
User-friendly interface that coordinates all components.

```python
class PlotModel:
    def __init__(self, data: Sequence[Sequence[Union[str, int, float]]])
    @property
    def matrix(self) -> Matrix
    @property
    def tick_manager(self) -> AxisTickManager
    def append_data(self, row: Sequence[Union[str, int, float]])
    def remove_data(self, index: int)
```

## Normalization Strategy

### Data Views

Normalization is provided through lightweight views:

```python
class NormalizedNumericalArrayView:
    def __init__(self, array: NumericalArray, min_val: float = 0, max_val: float = 1):
        self._array = array
    
    @property
    def normalized_values(self) -> NDArray[float]:
        """Compute normalized values using numpy for performance"""
        # Cast to numpy array for efficient computation
        values = np.array(self._array.values, dtype=np.float64)
        return (values - self._array.min) / (self._array.max - self._array.min)
```

### Matrix Views

```python
class NormalizedMatrixView:
    def __init__(self, matrix: Matrix, normalization_ranges: Optional[Dict[int, Tuple[float, float]]] = None):
        self._matrix = matrix
        self._normalization_ranges = normalization_ranges or {}
        self._column_views = {}
    
    def get_normalized_column(self, index: int) -> NormalizedArrayView:
        if index not in self._column_views:
            array = self._matrix.get_column(index)
            min_val, max_val = self._normalization_ranges.get(index, (0, 1))
            self._column_views[index] = NormalizedArrayView(array, min_val, max_val)
        return self._column_views[index]
```

## Renderer System

### Base Renderer Interface

```python
class BaseRenderer(ABC):
    def __init__(self):
        
    @abstractmethod
    def show
```


### Matplotlib Renderer

```python
class MatplotlibRenderer(BaseRenderer):
    def __init__(self):
        super().__init__()
    
    def render(self, normalized_view: NormalizedMatrixView):
        """Render the plot with fresh normalization data"""
        self._figure, self._axes = plt.subplots()
        self._draw_plot(normalized_view)
        return self._figure
    
    def _draw_plot(self, normalized_view: NormalizedMatrixView):
        """Draw the plot using the provided normalized data"""
        # Get normalized data as needed during rendering
        for col_idx in range(normalized_view.num_columns):
            normalized_col = normalized_view.get_normalized_column(col_idx)
            values = normalized_col.normalized_values
            
            # Apply customizations
            color = self._customizations.get('color', 'blue')
            linewidth = self._customizations.get('linewidth', 1.0)
            
            # Draw the lines
            self._axes.plot([col_idx] * len(values), values, 
                          color=color, linewidth=linewidth)
        
        # Apply tick customizations
        if 'ticks' in self._customizations:
            for col_idx, ticks in self._customizations['ticks'].items():
                self._axes.set_xticks([col_idx])
                self._axes.set_xticklabels([str(t) for t in ticks])
```

## Main Interface

### `PaxPlot`

The main user interface that coordinates everything:

```python
class PaxPlot:
    def __init__(self):
        self._plot_model: Optional[PlotModel] = None
        self._renderers: Dict[str, BaseRenderer] = {}
        self._customizations = {}  # Store customization settings
    
    def plot(self, data: Sequence[Sequence[Union[str, int, float]]]):
        """Set up plot data"""
        self._plot_model = PlotModel(data)
        return self
    
    def remove_row(self, index: int):
        """Remove a row from the plot data"""
        if self._plot_model:
            self._plot_model.remove_row(index)
        return self
    
    def add_renderer(self, name: str, renderer: BaseRenderer):
        """Add a renderer"""
        self._renderers[name] = renderer
        return self
    
    def set_color(self, color: str):
        """Customize plot color"""
        self._customizations['color'] = color
        return self
    
    def set_linewidth(self, width: float):
        """Customize line width"""
        self._customizations['linewidth'] = width
        return self
    
    def set_ticks(self, column: int, ticks: List[Union[float, str]]):
        """Customize ticks for a column"""
        if 'ticks' not in self._customizations:
            self._customizations['ticks'] = {}
        self._customizations['ticks'][column] = ticks
        return self
    
    def show(self):
        """Render and display all plots"""
        if not self._plot_model:
            raise ValueError("No data to plot. Call plot() first.")
        
        results = {}
        for name, renderer in self._renderers.items():
            # Create normalization views as needed during rendering
            normalized_view = NormalizedMatrixView(
                self._plot_model.matrix,
                normalization_ranges={i: (0, 1) for i in range(self._plot_model.num_columns)}
            )
            
            # Apply customizations
            renderer.apply_customizations(self._customizations)
            
            # Render with fresh normalization
            results[name] = renderer.render(normalized_view)
        
        return results
```

## Usage Examples

### Basic Usage

```python
# Initialize, plot, customize, show
plot = PaxPlot()
plot.plot(data)
plot.show()  # Renders with fresh normalization
```

### Multiple Renderers

```python
# Create plot with multiple renderers
plot = PaxPlot()
plot.add_renderer('matplotlib', MatplotlibRenderer())
plot.add_renderer('plotly', PlotlyRenderer())

# All renderers show when show() is called
plot.plot(data)
plot.show()  # Both matplotlib and plotly render
```

### Customization

```python
# Customize before showing
plot = PaxPlot()
plot.plot(data)
plot.set_color('red')
plot.set_linewidth(2.0)
plot.set_ticks(column=0, ticks=[1, 2, 3, 4])
plot.show()  # Renders with all customizations
```

### Simple Normalization

```python
# Normalization happens during show()
plot = PaxPlot()
plot.plot(data)

# No normalization computed yet
# ... customize plot ...

plot.show()  # Normalization computed during rendering
# All renderers get fresh normalized data
```

## Benefits

1. **Separation of Concerns**: Data, normalization, and rendering are independent
2. **Simple and Predictable**: No automatic re-rendering, users control when to plot
3. **Renderer Agnostic**: Easy to add new rendering backends
4. **Simple Architecture**: No complex caching or notification systems
5. **Maintainability**: Clear boundaries between components
6. **Testability**: Each component can be tested independently
7. **Extensibility**: Easy to add new features without affecting existing code
8. **Memory Efficient**: No observer or caching overhead
9. **Industry Standard**: Similar to matplotlib's approach
10. **Easy to Understand**: Straightforward data flow and rendering

## Migration Strategy

1. Implement core data structures (`NumericalArray`, `CategoricalArray`, `Matrix`)
2. Create tick and limit management classes
3. Implement normalization views
4. Build `PlotModel` and `PaxPlot` interfaces
5. Create renderer system with base interface
6. Migrate existing matplotlib functionality to new renderer
7. Add tests for all components
8. Update documentation and examples

This architecture provides a solid foundation for PaxPlot's future development while maintaining clean separation between data management, normalization, and rendering concerns.
