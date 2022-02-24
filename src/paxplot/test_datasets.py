"""Tests for paxplot dataset"""

import io
import unittest
import datasets


class PaxplotDatasets(unittest.TestCase):
    def test_tradeoff(self):
        """
        Test for tradeoff dataset
        """
        stream = datasets.tradeoff()
        self.assertIsInstance(
            stream,
            io.BufferedReader
        )
        stream.close()


if __name__ == '__main__':
    unittest.main()
