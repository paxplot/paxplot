"""Tests for core paxplot functions"""

import unittest

import core


class AnalysisLib(unittest.TestCase):
    def test_parallel_blank(self):
        """
        Test of blank figure
        """
        # Run
        paxfig, paxes = core.pax_parallel(n_axes=4)

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
        for ax in paxes.axes:
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
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]

        # Run
        paxfig, paxes = core.pax_parallel(n_axes=len(data[0]))
        paxes.plot(data)

        # Test plotted data
        self.assertTrue(
            (paxes.axes[0].lines[0].get_ydata() == [0.0, 0.0]).all()
        )
        self.assertTrue(
            (paxes.axes[0].lines[1].get_ydata() == [0.5, 0.5]).all()
        )
        self.assertTrue(
            (paxes.axes[0].lines[2].get_ydata() == [1.0, 1.0]).all()
        )

        # Test ticks
        self.assertEqual(paxes.axes[0].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(
            paxes.axes[0].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(paxes.axes[0].get_yticklabels()[-1].get_text(), '2.0')
        self.assertEqual(
            paxes.axes[0].get_yticklabels()[-1].get_position()[1], 1.0
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
        paxfig, paxes = core.pax_parallel(n_axes=len(data[0]))
        paxes.plot(data)
        paxes.set_ylim(paxes.axes[0], bottom=-1.0, top=3)
        paxes.set_ylim(paxes.axes[2], bottom=0.0, top=1.5)

        # Test plotted data
        self.assertEqual(paxes.axes[0].lines[0].get_ydata()[0], 0.25)
        self.assertEqual(paxes.axes[0].lines[1].get_ydata()[0], 0.50)
        self.assertEqual(paxes.axes[0].lines[2].get_ydata()[0], 0.75)
        self.assertEqual(paxes.axes[1].lines[0].get_ydata()[1], 0.0, 1)
        self.assertAlmostEqual(
            paxes.axes[1].lines[1].get_ydata()[1], 0.6666, 2
        )
        self.assertAlmostEqual(
            paxes.axes[1].lines[2].get_ydata()[1], 1.3333, 2
        )

        # Test ticks
        self.assertEqual(paxes.axes[0].get_yticklabels()[0].get_text(), '-1.0')
        self.assertEqual(
            paxes.axes[0].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(paxes.axes[0].get_yticklabels()[-1].get_text(), '3.0')
        self.assertEqual(
            paxes.axes[0].get_yticklabels()[-1].get_position()[1], 1.0
        )
        self.assertEqual(paxes.axes[2].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(
            paxes.axes[2].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(paxes.axes[2].get_yticklabels()[-1].get_text(), '1.5')
        self.assertEqual(
            paxes.axes[2].get_yticklabels()[-1].get_position()[1], 1.0
        )

    def test_parallel_yticks(self):
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
        paxfig, paxes = core.pax_parallel(n_axes=len(data[0]))
        paxes.plot(data)
        paxes.set_yticks(paxes.axes[0], ticks=[0.0, 1.0, 2.0])
        paxes.set_yticks(paxes.axes[1], ticks=[0.0, 0.5, 1.0, 2.0, 2.5])

        # Test ticks
        self.assertEqual(paxes.axes[0].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(paxes.axes[0].get_yticklabels()[2].get_text(), '2.0')
        self.assertEqual(paxes.axes[1].get_yticklabels()[0].get_text(), '0.0')
        self.assertEqual(paxes.axes[1].get_yticklabels()[4].get_text(), '2.5')
        self.assertEqual(
            paxes.axes[0].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxes.axes[0].get_yticklabels()[2].get_position()[1], 1.0
        )
        self.assertEqual(
            paxes.axes[1].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxes.axes[1].get_yticklabels()[4].get_position()[1], 1.0
        )

    def test_parallel_custom_yticks(self):
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
        paxfig, paxes = core.pax_parallel(n_axes=len(data[0]))
        paxes.plot(data)
        paxes.set_yticks(
            paxes.axes[0],
            ticks=[0.0, 1.0, 2.0],
            labels=['my heart', 'is the \ncode to', '1612']
        )

        # Test ticks
        self.assertEqual(
            paxes.axes[0].get_yticklabels()[0].get_text(), 'my heart'
        )
        self.assertEqual(paxes.axes[0].get_yticklabels()[2].get_text(), '1612')
        self.assertEqual(
            paxes.axes[0].get_yticklabels()[0].get_position()[1], 0.0
        )
        self.assertEqual(
            paxes.axes[0].get_yticklabels()[2].get_position()[1], 1.0
        )

    def test_parallel_xlabel(self):
        # Setup
        data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ]

        # Run
        paxfig, paxes = core.pax_parallel(n_axes=len(data[0]))
        paxes.plot(data)
        paxes.set_xlabel(
            paxes.axes[0],
            xlabel='Em7-G7-Cmaj7'
        )
        paxes.set_xlabel(
            paxes.axes[1],
            xlabel='F#m7-B7'
        )

        # Test
        self.assertEqual(
            paxes.axes[0].get_xticklabels()[0].get_text(), 'Em7-G7-Cmaj7'
        )
        self.assertEqual(
            paxes.axes[1].get_xticklabels()[0].get_text(), 'F#m7-B7'
        )


if __name__ == '__main__':
    unittest.main()
