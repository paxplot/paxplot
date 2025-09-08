"""Axis label dataclass for storing axis label information."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AxisLabel:
    """Dataclass for storing axis label information.

    This class provides a simple container for axis label data,
    primarily storing a string label that can be None by default.

    Parameters
    ----------
    label : Optional[str], default=None
        The text label for the axis.

    Examples
    --------
    >>> axis_label = AxisLabel("Temperature (°C)")
    >>> print(axis_label.label)
    Temperature (°C)

    >>> empty_label = AxisLabel()
    >>> print(empty_label.label)
    None
    """

    label: Optional[str] = None
