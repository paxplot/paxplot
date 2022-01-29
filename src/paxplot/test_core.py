"""Tests for core paxplot functions"""

from tokenize import PlainToken
import unittest

import matplotlib
from matplotlib.pyplot import plot

import core


class PaxplotLib(unittest.TestCase):
    def test_parallel_blank(self):
        """
        Test of blank figure
        """
        # Run
        paxfig = core.pax_parallel(n_axes=4)

        # Test
        self.assertEqual(paxfig.axes.__len__(), 4)
        self.assertEqual(
            paxfig.subplotpars.__getattribute__('wspace'),
            0.0
        )
        self.assertEqual(
            paxfig.subplotpars.__getattribute__('hspace'),
            0.0
        )
        for ax in paxfig.axes:
            self.assertEqual(
                ax.spines['top'].get_visible(),
                False
            ),
            self.assertEqual(
                ax.spines['bottom'].get_visible(),
                False
            )
            self.assertEqual(
                ax.spines['right'].get_visible(),
                False
            )
            self.assertEqual(ax.get_ylim(), (0.0, 1.0))
            self.assertEqual(ax.get_xlim(), (0.0, 1.0))
            self.assertEqual(ax.get_xticks(), [0])
            self.assertEqual(
                ax.xaxis.majorTicks[0].tick1line.get_markersize(),
                0.0
            )

        # Last axis
        position = paxfig.axes[-1].get_position()
        width = position.x1 - position.x0
        self.assertEqual(width, 0.0)

    def test_parallel_basic(self):
        """
        Basic test of parallel functionality
        """
        # Setup
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]

        # Run
        paxfig = core.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Test plotted data
        self.assertTrue(
            (paxfig.axes[0].lines[0].get_ydata() == [0.0, 0.0]).all()
        )
        self.assertTrue(
            (paxfig.axes[0].lines[1].get_ydata() == [0.5, 0.5]).all()
        )
        self.assertTrue(
            (paxfig.axes[0].lines[2].get_ydata() == [1.0, 1.0]).all()
        )

        # Test ticks
        self.assertEqual(paxfig.axes[0].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_text(), '2.0'
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_position()[1], 1.0
        )

    def test_parallel_limits(self):
        """
        Testing plotting with custom limits
        """
        # Setup
        data = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ]

        # Run
        paxfig = core.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.set_lim(ax_idx=0, bottom=-1.0, top=3)
        paxfig.set_lim(ax_idx=2, bottom=0.0, top=1.5)

        # Test plotted data
        self.assertEqual(paxfig.axes[0].lines[0].get_ydata()[0], 0.25)
        self.assertEqual(paxfig.axes[0].lines[1].get_ydata()[0], 0.50)
        self.assertEqual(paxfig.axes[0].lines[2].get_ydata()[0], 0.75)
        self.assertEqual(paxfig.axes[1].lines[0].get_ydata()[1], 0.0, 1)
        self.assertAlmostEqual(
            paxfig.axes[1].lines[1].get_ydata()[1], 0.6666, 2
        )
        self.assertAlmostEqual(
            paxfig.axes[1].lines[2].get_ydata()[1], 1.3333, 2
        )

        # Test ticks
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[0].get_text(), '-1.0'
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_text(), '3.0'
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_position()[1], 1.0
        )
        self.assertEqual(paxfig.axes[2].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[-1].get_text(), '1.5'
        )
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[-1].get_position()[1], 1.0
        )

    def test_parallel_ticks(self):
        """
        Testing plotting with modified ticks
        """
        # Setup
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]

        # Run
        paxfig = core.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.set_ticks(ax_idx=0, ticks=[0.0, 1.0, 2.0])
        paxfig.set_ticks(ax_idx=1, ticks=[0.0, 0.5, 1.0, 2.0, 2.5])

        # Test ticks
        self.assertEqual(paxfig.axes[0].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(paxfig.axes[0].get_yticklabels()[2].get_text(), '2.0')
        self.assertEqual(paxfig.axes[1].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(paxfig.axes[1].get_yticklabels()[4].get_text(), '2.5')
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[2].get_position()[1], 1.0
        )
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[4].get_position()[1], 1.0
        )

    def test_parallel_custom_ticks(self):
        """
        Testing plotting with custom ticks
        """
        # Setup
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]

        # Run
        paxfig = core.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.set_ticks(
            ax_idx=0,
            ticks=[0.0, 1.0, 2.0],
            labels=['my heart', 'is the \ncode to', '1612']
        )

        # Test ticks
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[0].get_text(), 'my heart'
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[2].get_text(), '1612'
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[2].get_position()[1], 1.0
        )

    def test_parallel_label(self):
        """
        Testing setting labels
        """
        # Setup
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]

        # Run
        paxfig = core.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.set_label(
            ax_idx=0,
            label='Em7-Dm7-G7'
        )
        paxfig.set_label(
            ax_idx=1,
            label='Cmaj7-B7'
        )

        # Test
        self.assertEqual(
            paxfig.axes[0].get_xticklabels()[0].get_text(), 'Em7-Dm7-G7'
        )
        self.assertEqual(
            paxfig.axes[1].get_xticklabels()[0].get_text(), 'Cmaj7-B7'
        )
        import matplotlib.pyplot as plt
        plt.show()

    def test_parallel_invert(self):
        """
        Test inverting axis
        """
        # Setup
        data = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ]

        # Run
        paxfig = core.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.invert_axis(ax_idx=0)
        paxfig.invert_axis(ax_idx=2)

        # Test plotted data
        self.assertEqual(paxfig.axes[0].lines[0].get_ydata()[0], 1.0)
        self.assertEqual(paxfig.axes[0].lines[2].get_ydata()[0], 0.0)
        self.assertEqual(paxfig.axes[1].lines[0].get_ydata()[1], 1.0)
        self.assertEqual(paxfig.axes[1].lines[2].get_ydata()[1], 0.0)

        # Test ticks
        self.assertEqual(paxfig.axes[0].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_text(), '2.0'
        )
        self.assertEqual(paxfig.axes[2].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[-1].get_text(), '2.0'
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[0].get_position()[1], 1.0
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[0].get_position()[1], 1.0
        )
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[-1].get_position()[1], 0.0
        )

    def test_parallel_legend(self):
        """
        Test creating legend
        """
        # Setup
        data = [
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 0.0],
        ]

        # Run
        paxfig = core.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.add_legend(label=['A', 'B', 'C'])

        # Legend tests
        legend_text = paxfig.axes[-1].get_legend().get_texts()
        self.assertEqual(legend_text[0].get_text(), 'A')
        self.assertEqual(legend_text[1].get_text(), 'B')
        self.assertEqual(legend_text[2].get_text(), 'C')

        # Gridspec tests
        self.assertEqual(
            paxfig.axes[0].get_gridspec().get_width_ratios(),
            [1.0, 1.0, 0.0, 1.0]
        )
        import matplotlib.pyplot as plt
        plt.show()

    def test_parallel_colorbar(self):
        """
        Test creating colorbar
        """
        # Setup
        data = [
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [3.0, 2.0, 0.0],
        ]

        # Run
        paxfig, paxes = core.pax_parallel(n_axes=len(data[0]))
        paxes.plot(data)
        paxfig.add_colorbar(ax=2, data=data, cmap='viridis')

        # Line color tests
        self.assertEqual(
            paxfig.axes[0].lines[0].get_color(),
            '#fde725'
        )
        self.assertEqual(
            paxfig.axes[0].lines[1].get_color(),
            '#21918c'
        )

        # Colorbar tests
        self.assertEqual(
            paxfig.axes[-1].get_label(),
            '<colorbar>'
        )
        self.assertEqual(
            paxfig.axes[-1].get_ylim(),
            (0.0, 2.0)
        )

        # Gridspec tests
        self.assertEqual(
            paxfig.axes[0].get_gridspec().get_width_ratios(),
            [1.0, 1.0, 0.0, 0.5]
        )


if __name__ == '__main__':
    unittest.main()
