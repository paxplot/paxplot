"""Tests for core paxplot functions"""

import unittest
import os

import core


class AnalysisLib(unittest.TestCase):
    def test_parallel_blank(self):
        """
        Test of blank figure
        """
        # Run
        fig, axes = core.pax_parallel(n_axes=4)

        # Test
        fig.show()
        self.assertEqual(fig.axes.__len__(), 4)
        self.assertEqual(
            fig.subplotpars.__getattribute__('wspace'),
            0.0
        )
        self.assertEqual(
            fig.subplotpars.__getattribute__('hspace'),
            0.0
        )
        for ax in fig.axes:
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

    def test_parallel_basic(self):
        """
        Basic test of parallel functionality
        """
        # Setup
        data = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0]
        ]

        # Run
        fig, axes = core.pax_parallel(n_axes=len(data[0]))
        axes.plot(data)

        # Test
        fig.show()


if __name__ == '__main__':
    unittest.main()
