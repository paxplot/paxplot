"""Tests for core paxplot functions"""

import unittest
import os
import random
from string import ascii_lowercase
import numpy as np
import matplotlib.pyplot as plt

import core


class AnalysisLib(unittest.TestCase):
    def test_parallel_blank(self):
        """
        Test of blank figure
        """
        # Run
        fig, axes = core.pax_parallel(n_axes=4)
        fig.savefig('temp.png')

        # Test
        image = open('temp.png', 'rb')
        image_test = open('test_solution_images/test_parallel_blank.png', 'rb')
        self.assertEqual(image.read(), image_test.read())

        # Cleanup
        image.close()
        image_test.close()
        os.remove('temp.png')
        

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
        fig.savefig('temp.png')

        # Test
        image = open('temp.png', 'rb')
        image_test = open('test_solution_images/test_parallel_basic.png', 'rb')
        self.assertEqual(image.read(), image_test.read())

        # Cleanup
        image.close()
        image_test.close()
        os.remove('temp.png')


if __name__ == '__main__':
    unittest.main()
