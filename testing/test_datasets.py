"""Tests for paxplot dataset"""

import io
import unittest
import paxplot


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

    def test_model_evaluation(self):
        """
        Test for model_evaluation dataset
        """
        stream = paxplot.model_evaluation()
        self.assertIsInstance(
            stream,
            io.BufferedReader
        )
        stream.close()


if __name__ == '__main__':
    unittest.main()
