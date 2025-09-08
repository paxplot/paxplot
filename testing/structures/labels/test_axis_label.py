"""Tests for AxisLabel dataclass."""

from paxplot.structures.labels import AxisLabel


class TestAxisLabel:
    """Test cases for AxisLabel dataclass."""

    def test_default_initialization(self):
        """Test that AxisLabel initializes with None by default."""
        label = AxisLabel()
        assert label.label is None

    def test_string_label_initialization(self):
        """Test that AxisLabel can be initialized with a string label."""
        label_text = "Temperature (°C)"
        label = AxisLabel(label=label_text)
        assert label.label == label_text

    def test_empty_string_label(self):
        """Test that AxisLabel can be initialized with an empty string."""
        label = AxisLabel(label="")
        assert label.label == ""

    def test_none_label_explicit(self):
        """Test that AxisLabel can be explicitly initialized with None."""
        label = AxisLabel(label=None)
        assert label.label is None

    def test_label_modification(self):
        """Test that the label can be modified after initialization."""
        label = AxisLabel()
        assert label.label is None

        label.label = "New Label"
        assert label.label == "New Label"

        label.label = None
        assert label.label is None

    def test_equality(self):
        """Test that AxisLabel instances with same label are equal."""
        label1 = AxisLabel("Test Label")
        label2 = AxisLabel("Test Label")
        assert label1 == label2

    def test_inequality(self):
        """Test that AxisLabel instances with different labels are not equal."""
        label1 = AxisLabel("Label 1")
        label2 = AxisLabel("Label 2")
        assert label1 != label2

    def test_equality_with_none(self):
        """Test equality when both labels are None."""
        label1 = AxisLabel()
        label2 = AxisLabel(label=None)
        assert label1 == label2

    def test_repr(self):
        """Test that the string representation is correct."""
        label = AxisLabel("Test Label")
        repr_str = repr(label)
        assert "AxisLabel" in repr_str
        assert "Test Label" in repr_str

    def test_repr_with_none(self):
        """Test string representation with None label."""
        label = AxisLabel()
        repr_str = repr(label)
        assert "AxisLabel" in repr_str
        assert "None" in repr_str
