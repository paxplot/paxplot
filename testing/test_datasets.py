"""Tests for paxplot dataset"""

import io
import unittest
import src as paxplot


class PaxplotDatasets(unittest.TestCase):
    def test_tradeoff(self):
        """
        Test for tradeoff dataset
        """
        stream = paxplot.tradeoff()
        self.assertIsInstance(
            stream,
            io.BufferedReader
        )
        stream.close()


if __name__ == '__main__':
    unittest.main()
