"""Tests for paxplot paxplot functions"""

import os
import unittest
import warnings
import matplotlib.pyplot as plt
import numpy as np
import src.paxplot as paxplot


class PaxplotLib(unittest.TestCase):
    def test_parallel_blank(self):
        """
        Test of blank figure
        """
        # Run
        paxfig = paxplot.pax_parallel(n_axes=4)

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
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Test attribute data
        self.assertEqual(
            paxfig._pax_lims,
            [[0.0, 2.0], [0.0, 2.0]]
        )

        # Test plotted data
        self.assertTrue(
            paxfig.axes[0].lines[0].get_ydata() == [0.0, 0.0]
        )
        self.assertTrue(
            paxfig.axes[0].lines[1].get_ydata() == [0.5, 0.5]
        )
        self.assertTrue(
            paxfig.axes[0].lines[2].get_ydata() == [1.0, 1.0]
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
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.set_lim(ax_idx=0, bottom=-1.0, top=3)
        paxfig.set_lim(ax_idx=1, bottom=-1.0, top=3)
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
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[0].get_text(), '-1.0'
        )
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[-1].get_text(), '3.0'
        )
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[-1].get_position()[1], 1.0
        )
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[0].get_text(), '0.0'
        )
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
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 2.0, 0.0, 3.0],
        ]

        # Run
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        tick_lists = [
            [0.0, 1.0, 3.0],
            [0.0, 1.0, 2.5],
            [-1.0, 1.0, 2.0],
            [0.0, 1.0, 3.5]
        ]
        for i, ticks in enumerate(tick_lists):
            paxfig.set_ticks(ax_idx=i, ticks=ticks)

        # Test tick labels
        for i, ax in enumerate(paxfig.axes):
            for j, label in enumerate(ax.get_yticklabels()):
                self.assertEqual(
                    label.get_text(),
                    str(tick_lists[i][j])
                )

        # Test tick positioning (random)
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[2].get_position()[1], 1.0
        )
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[1].get_position()[1], 0.4
        )
        self.assertAlmostEqual(
            paxfig.axes[2].get_yticklabels()[1].get_position()[1],
            0.6666,
            2
        )
        self.assertAlmostEqual(
            paxfig.axes[3].get_yticklabels()[1].get_position()[1],
            0.28571,
            2
        )

    def test_parallel_even_ticks(self):
        """
        Test even tick setting
        """
        # Setup
        data = [
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 2.0, 0.0, 3.0],
        ]

        # Run
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.set_even_ticks(
            ax_idx=0,
            n_ticks=30,
        )

        # Test tick position
        for i in range(31):
            self.assertAlmostEqual(
                paxfig.axes[0].get_yticklabels()[i].get_position()[1],
                i*0.03333,
                3
            )

        # Test tick labels
        for i in range(31):
            self.assertEqual(
                paxfig.axes[0].get_yticklabels()[i].get_text(),
                str(round(i*0.1, 2)),
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
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
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
        Testing setting label
        """
        # Setup
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]

        # Run
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
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

    def test_parallel_labels(self):
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
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.set_labels(labels=['A', 'B'])

        # Test
        self.assertEqual(
            paxfig.axes[0].get_xticklabels()[0].get_text(), 'A'
        )
        self.assertEqual(
            paxfig.axes[1].get_xticklabels()[0].get_text(), 'B'
        )

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
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.invert_axis(ax_idx=0)
        paxfig.invert_axis(ax_idx=2)

        # Test plotted data
        self.assertEqual(paxfig.axes[0].lines[0].get_ydata()[0], 1.0)
        self.assertEqual(paxfig.axes[0].lines[2].get_ydata()[0], 0.0)
        self.assertEqual(paxfig.axes[1].lines[0].get_ydata()[1], 1.0)
        self.assertEqual(paxfig.axes[1].lines[2].get_ydata()[1], 0.0)

        # Test ticks
        self.assertEqual(paxfig.axes[0].get_yticklabels()[0].get_text(), '2.0')
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_text(), '0.0'
        )
        self.assertEqual(paxfig.axes[2].get_yticklabels()[0].get_text(), '2.0')
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[-1].get_text(), '0.0'
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_position()[1], 1.0
        )
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[-1].get_position()[1], 1.0
        )

    def test_parallel_invert_middle(self):
        """
        Test inverting middle axes
        """
        # Setup
        data = [
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [3.0, 2.0, 0.0],
        ]

        # Run
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.invert_axis(1)

        # Test plotted data
        self.assertEqual(paxfig.axes[0].lines[0].get_ydata()[0], 0.0)
        self.assertEqual(paxfig.axes[0].lines[2].get_ydata()[0], 1.0)
        self.assertEqual(paxfig.axes[1].lines[0].get_ydata()[1], 1.0)
        self.assertEqual(paxfig.axes[1].lines[2].get_ydata()[1], 0.0)

        # Test ticks
        self.assertEqual(paxfig.axes[0].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_text(), '3.0'
        )
        self.assertEqual(paxfig.axes[1].get_yticklabels()[0].get_text(), '2.0')
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[-1].get_text(), '0.0'
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_position()[1], 1.0
        )
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[0].get_position()[0], 0.0
        )
        self.assertEqual(
            paxfig.axes[2].get_yticklabels()[-1].get_position()[1], 1.0
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
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.add_legend(labels=['A', 'B', 'C'])

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
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.add_colorbar(
            ax_idx=2,
            cmap='viridis',
            colorbar_kwargs={'label': 'test'}
        )

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
            paxfig.axes[-1].get_ylabel(),
            'test'
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

    def test_parallel_combo(self):
        """
        Test applying multiple functions. This is more of an
        integration test that can quickly check functionality
        the many methods working together.
        """
        # Setup
        data = [
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [3.0, 2.0, 0.0],
        ]
        # Run
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)
        paxfig.set_lim(0, bottom=0, top=4)
        paxfig.set_lim(1, bottom=0, top=4)
        paxfig.set_lim(2, bottom=0, top=5)
        paxfig.set_ticks(0, [0, 1, 2, 3])
        paxfig.set_ticks(1, [0, 0.5, 2.0], ['a 0', 'b 0.5', 'c 2.0'])
        paxfig.set_ticks(2, [-1.0, 0.0, 0.5, 1.0, 2.0, 5.0])
        paxfig.set_label(0, 'foo')
        paxfig.invert_axis(1)
        paxfig.add_colorbar(0)

        # Y data tests (random)
        self.assertEqual(paxfig.axes[0].lines[1].get_ydata()[0], 0.25)
        self.assertEqual(paxfig.axes[1].lines[2].get_ydata()[0], 0.50)

        # Tick label tests (random)
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[0].get_text(),
            '0'
        )
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[1].get_text(),
            'b 0.5'
        )

        # Tick positioning tests (random)
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[-1].get_position()[1],
            0.5
        )
        self.assertEqual(
            paxfig.axes[0].get_yticklabels()[-1].get_position()[1],
            0.75
        )

        # Label test
        self.assertEqual(
            paxfig.axes[0].get_xticklabels()[0].get_text(), 'foo'
        )

        # Colorbar tests
        self.assertEqual(
            paxfig.axes[-1].get_label(),
            '<colorbar>'
        )
        self.assertEqual(
            paxfig.axes[-1].get_ylim(),
            (0.0, 4.0)
        )

    def test_parallel_singleton(self):
        """
        Testing if axis has all same points and still render plot
        """
        # Setup
        data = [
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
        # Run
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Y data tests
        self.assertEqual(paxfig.axes[1].lines[0].get_ydata()[0], 0.50)
        self.assertEqual(paxfig.axes[1].lines[1].get_ydata()[0], 0.50)
        self.assertEqual(paxfig.axes[1].lines[2].get_ydata()[0], 0.50)

        # Test tick labels
        self.assertEqual(paxfig.axes[1].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(
            paxfig.axes[1].get_yticklabels()[-1].get_text(), '2.0'
        )

    def test_parallel_savefig(self):
        """
        Testing if possible to save figure
        """
        # Run
        paxfig = paxplot.pax_parallel(n_axes=2)
        paxfig.savefig('test.png')

        # Test if file generated
        self.assertTrue(os.path.isfile('test.png'))

        # Cleanup
        os.remove('test.png')

    def test_parallel_not_supported(self):
        """
        Testing if unsupported
        """
        # Run
        paxfig = paxplot.pax_parallel(n_axes=2)

        # Test
        with self.assertWarns(Warning):
            paxfig.suptitle('Test')

    def test_parallel_supported_top_level(self):
        """
        Testing if top_level functions raise warnings
        """
        # Setup
        paxfig = paxplot.pax_parallel(n_axes=2)

        with warnings.catch_warnings(record=True) as w:
            # Run
            plt.show(block=False)

            # Test
            self.assertEqual(len(w), 0)

    def test_parallel_figshow(self):
        """
        Testing if .show raises warnings
        """
        # Setup
        paxfig = paxplot.pax_parallel(n_axes=2)

        with warnings.catch_warnings(record=True) as w:
            # Run
            paxfig.show()

            # Test
            self.assertEqual(len(w), 0)

    def test_multi_plot(self):
        """
        Calling plot multiple times
        """
        # Run
        paxfig = paxplot.pax_parallel(n_axes=3)
        paxfig.plot(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0]
            ]
        )
        paxfig.plot(
            [
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )

        # Test Attributes
        self.assertEqual(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0]
            ],
            paxfig._pax_data.tolist()
        )
        np.testing.assert_almost_equal(
            [
                [0.0, 0.0, 0.0],
                [0.33, 0.33, 0.33],
                [0.66, 0.66, 0.66],
                [1.0, 1.0, 1.0]
            ],
            paxfig._pax_data_scale.tolist(),
            decimal=1
        )

    def test_custom_multi_plot(self):
        """
        Multiple calls to plot with custom ticks and limits
        """
        # Run
        paxfig = paxplot.pax_parallel(n_axes=3)
        paxfig.plot(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0]
            ]
        )
        paxfig.set_lim(ax_idx=0, bottom=0.0, top=5.0)
        paxfig.set_ticks(ax_idx=1, ticks=[0.0, 1.0, 2.0])
        paxfig.plot(
            [
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )

        # Test Attributes
        self.assertEqual(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0]
            ],
            paxfig._pax_data.tolist()
        )
        np.testing.assert_almost_equal(
            [
                [0.2, 0.25, 0.0],
                [0.4, 0.5, 0.33],
                [0.6, 0.75, 0.66],
                [0.8, 1.0, 1.0]
            ],
            paxfig._pax_data_scale.tolist(),
            decimal=1
        )
        self.assertEqual(
            paxfig._pax_ticks[1],
            [0.0, 1.0, 2.0]
        )

    def test_attributes_plot(self):
        """
        Calling plot with attrubutes
        """
        # Run
        paxfig = paxplot.pax_parallel(n_axes=3)
        paxfig.plot(
            data=[
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0]
            ],
            line_kwargs={
                'label': ['A', 'B'],
                'alpha': 0.5
            }
        )
        paxfig.add_legend()

        # Test
        self.assertEqual(
            paxfig.axes[0].lines[0].get_alpha(),
            0.5
        )
        self.assertEqual(
            paxfig.axes[0].lines[0].get_label(),
            'A'
        )
        self.assertEqual(
            paxfig.axes[0].lines[1].get_label(),
            'B'
        )


class PaxplotException(unittest.TestCase):
    def test_paxfig_creation(self):
        """
        Various ways to fail figure creation
        """
        # Nothing supplied
        with self.assertRaises(TypeError):
            paxplot.pax_parallel()

        # Non-int n_axes
        with self.assertRaises(TypeError):
            paxplot.pax_parallel(n_axes=0.1)

    def test_plot(self):
        """
        Various ways PaxFigure.plot can fail
        """
        # Too little data supplied
        with self.assertWarns(Warning):
            paxfig = paxplot.pax_parallel(n_axes=4)
            paxfig.plot(
                [
                    [0.0, 0.0, 2.0],
                    [1.0, 1.0, 1.0],
                ]
            )

        # Too much data supplied
        with self.assertRaises(ValueError):
            paxfig = paxplot.pax_parallel(n_axes=2)
            paxfig.plot(
                [
                    [0.0, 0.0, 2.0],
                    [1.0, 1.0, 1.0],
                ]
            )

        # Non-plottable data
        with self.assertRaises(TypeError):
            paxfig = paxplot.pax_parallel(n_axes=3)
            paxfig.plot(
                [
                    ['foo', 0.0, 2.0],
                    ['bar', 1.0, 1.0],
                ]
            )

    def test_lim(self):
        """
        Various ways PaxFigure.set_lim can fail
        """
        # Setup
        data = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ]
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Requesting axis that doesn't exist
        with self.assertRaises(IndexError):
            paxfig.set_lim(ax_idx=4, bottom=-1.0, top=3)

        # Non integer value for ax_idx
        with self.assertRaises(TypeError):
            paxfig.set_lim(ax_idx='foo', bottom=0, top=1)

        # Non numeric value for bottom
        with self.assertRaises(TypeError):
            paxfig.set_lim(ax_idx=0, bottom='foo', top=1)

        # Non numeric value for top
        with self.assertRaises(TypeError):
            paxfig.set_lim(ax_idx=0, bottom=0, top='foo')

    def test_ticks(self):
        """
        Various ways PaxFigure.set_ticks can fail
        """
        # Setup
        data = [
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 2.0, 0.0, 3.0],
        ]
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Requesting axis that doesn't exist
        with self.assertRaises(IndexError):
            paxfig.set_ticks(ax_idx=5, ticks=[0, 1, 2])

        # Non integer value for ax_idx
        with self.assertRaises(TypeError):
            paxfig.set_ticks(ax_idx='foo', ticks=[0, 1, 2])

        # Ticks non array-like
        with self.assertRaises(TypeError):
            paxfig.set_ticks(ax_idx=0, ticks='foo')

        # Non-numeric ticks
        with self.assertRaises(TypeError):
            paxfig.set_ticks(ax_idx=0, ticks=[1, 2, 'three'])

        # Labels non array-like or different lengths
        with self.assertRaises(ValueError):
            paxfig.set_ticks(ax_idx=0, ticks=[1, 2, 3], labels='A')

    def test_even_ticks(self):
        """
        Various ways PaxFigure.set_even_ticks can fail
        """
        # Setup
        data = [
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 2.0, 0.0, 3.0],
        ]

        # Run
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Requesting axis that doesn't exist
        with self.assertRaises(IndexError):
            paxfig.set_even_ticks(
                ax_idx=5,
                n_ticks=30,
                minimum=0.0,
                maximum=3.0
            )

        # Non integer value for ax_idx
        with self.assertRaises(TypeError):
            paxfig.set_even_ticks(
                ax_idx='foo',
                n_ticks=30,
                minimum=0.0,
                maximum=3.0
            )

        # Non integer value for n_ticks
        with self.assertRaises(TypeError):
            paxfig.set_even_ticks(
                ax_idx=0,
                n_ticks='foo',
                minimum=0.0,
                maximum=3.0
            )

        # Maximum greater than minimum
        with self.assertRaises(ValueError):
            paxfig.set_even_ticks(
                ax_idx=0,
                n_ticks=10,
                minimum=3.0,
                maximum=0.0
            )

    def test_label(self):
        """
        Various ways PaxFigure.set_label can fail
        """
        # Setup
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Requesting axis that doesn't exist
        with self.assertRaises(IndexError):
            paxfig.set_label(ax_idx=3, label='foo')

        # Non integer value for ax_idx
        with self.assertRaises(TypeError):
            paxfig.set_label(ax_idx='foo', label='bar')

    def test_labels(self):
        """
        Various ways PaxFigure.set_labels can fail
        """
        # Setup
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Requesting too many labels
        with self.assertRaises(IndexError):
            paxfig.set_labels(['A', 'B', 'C'])

    def test_invert(self):
        """
        Various ways PaxFigure.invert_axis can fail
        """
        # Setup
        data = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ]
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Requesting axis that doesn't exist
        with self.assertRaises(IndexError):
            paxfig.invert_axis(ax_idx=3)

        # Non integer value for ax_idx
        with self.assertRaises(TypeError):
            paxfig.invert_axis(ax_idx='foo')

    def test_legend(self):
        """
        Various ways PaxFigure.add_legend can fail
        """
        # Setup
        data = [
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 0.0],
        ]
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Not enough labels provided
        with self.assertRaises(IndexError):
            paxfig.add_legend(labels=['A', 'B'])

        # Too many labels provided
        with self.assertWarns(Warning):
            paxfig.add_legend(labels=['A', 'B', 'C', 'D'])

    def test_colorbar(self):
        """
        Various ways PaxFigure.add_colorbar can fail
        """
        # Setup
        data = [
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [3.0, 2.0, 0.0],
        ]
        paxfig = paxplot.pax_parallel(n_axes=len(data[0]))
        paxfig.plot(data)

        # Requesting axis that doesn't exist
        with self.assertRaises(IndexError):
            paxfig.add_colorbar(
                ax_idx=4,
                cmap='viridis',
            )

        # Non integer value for ax_idx
        with self.assertRaises(TypeError):
            paxfig.add_colorbar(
                ax_idx='foo',
                cmap='viridis',
            )

        # Colorbar that doesn't exist (default message helpful enough)
        with self.assertRaises(ValueError):
            paxfig.add_colorbar(
                ax_idx=0,
                cmap='foo',
            )

    def test_no_data(self):
        """
        Various ways PaxFigure can fail if no data is plotted
        """
        # Setup
        paxfig = paxplot.pax_parallel(n_axes=3)

        # Setting limits not supported
        with self.assertRaises(AttributeError):
            paxfig.set_lim(ax_idx=0, bottom=-1.0, top=3.0)

        # Setting ticks
        with self.assertRaises(AttributeError):
            paxfig.set_ticks(ax_idx=0, ticks=[0.0, 1.0, 2.0])

        # Setting labels supported
        paxfig.set_label(ax_idx=1, label='foo')

        # axis inversion
        with self.assertRaises(AttributeError):
            paxfig.invert_axis(ax_idx=2)

        # Legend won't fail but just creates blank legend
        paxfig.add_legend(labels=[])

        # Adding colorbar supported
        paxfig.add_colorbar(
            ax_idx=0,
            cmap='viridis',
        )


if __name__ == '__main__':
    unittest.main()
