"""Custom axis limits for PaxPlot.

This module defines the CustomAxisLimit dataclass for storing
client-specified custom limits for axis display.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class CustomAxisLimit:
    """
    Client-specified custom limits for axis display.

    This dataclass stores custom min/max values that override the natural
    data bounds when displaying plots. It does not affect the underlying data.

    Parameters
    ----------
    min_val : Optional[float], default=None
        Custom minimum value for the axis. Must be numeric or None.
    max_val : Optional[float], default=None
        Custom maximum value for the axis. Must be numeric or None.

    Raises
    ------
    TypeError
        If min_val or max_val are not numeric or None.
    ValueError
        If both min_val and max_val are set and min_val >= max_val.
        If min_val or max_val are NaN or infinity.

    Examples
    --------
    >>> # Set both limits
    >>> limits = CustomAxisLimit(min_val=0.0, max_val=10.0)

    >>> # Set only minimum
    >>> limits = CustomAxisLimit(min_val=5.0)

    >>> # No limits (use natural data bounds)
    >>> limits = CustomAxisLimit()

    >>> # These will raise errors:
    >>> CustomAxisLimit(min_val="hello")  # TypeError
    >>> CustomAxisLimit(min_val=10.0, max_val=5.0)  # ValueError
    >>> CustomAxisLimit(min_val=float('nan'))  # ValueError
    """

    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def __post_init__(self):
        """Validate the limits after initialization."""
        self._validate_types()
        self._validate_values()

    def _validate_types(self):
        """Validate that values are numeric or None."""
        if self.min_val is not None and not isinstance(
            self.min_val, (int, float)
        ):
            raise TypeError(
                f"min_val must be numeric or None, got {type(self.min_val)}: {self.min_val}"
            )

        if self.max_val is not None and not isinstance(
            self.max_val, (int, float)
        ):
            raise TypeError(
                f"max_val must be numeric or None, got {type(self.max_val)}: {self.max_val}"
            )

    def _validate_values(self):
        """Validate that values are finite and min < max."""
        # Check for NaN and infinity
        if self.min_val is not None:
            if math.isnan(self.min_val):
                raise ValueError("min_val cannot be NaN")
            if math.isinf(self.min_val):
                raise ValueError("min_val cannot be infinity")

        if self.max_val is not None:
            if math.isnan(self.max_val):
                raise ValueError("max_val cannot be NaN")
            if math.isinf(self.max_val):
                raise ValueError("max_val cannot be infinity")

        # Check range if both values are set
        if self.min_val is not None and self.max_val is not None:
            if self.min_val >= self.max_val:
                raise ValueError(
                    f"min_val ({self.min_val}) must be less than max_val ({self.max_val})"
                )

    def is_set(self) -> bool:
        """
        Check if any custom limits are configured.

        Returns
        -------
        bool
            True if either min_val or max_val is set, False otherwise.

        Examples
        --------
        >>> limits = CustomAxisLimit()
        >>> limits.is_set()
        False

        >>> limits = CustomAxisLimit(min_val=0.0)
        >>> limits.is_set()
        True

        >>> limits = CustomAxisLimit(max_val=10.0)
        >>> limits.is_set()
        True
        """
        return self.min_val is not None or self.max_val is not None
