"""Tests for core parapy functions"""

import unittest

import core


class AnalysisLib(unittest.TestCase):
    def test_parallel_basic(self):
        """
        Basic test of parallel functionality
        """
        # Setup
        data = [
            {'A': 0.0, 'B': 0.0, 'C': 0.0},
            {'A': 1.0, 'B': 1.0, 'C': 1.0},
            {'A': 2.0, 'B': 2.0, 'C': 2.0},
        ]

        # Run
        fig = core.parallel(data=data, cols=['A', 'B', 'C'])


if __name__ == '__main__':
    unittest.main()
