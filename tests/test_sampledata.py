import unittest

import pandas as pd
from parameterized import parameterized

import torecsys.data.sample_data as sampledata


class SampleDataTestCase(unittest.TestCase):
    @parameterized.expand([
        ("20m", "./sampledata"),
        ("latest-small", "./sampledata"),
    ])
    def test_download_ml_data(self, size, directory):
        is_downloaded = sample_data.download_ml_data(size, directory=directory)
        self.assertEqual(is_downloaded, True)

    @parameterized.expand([
        ("20m", "./sampledata"),
        ("latest-small", "./sampledata"),
    ])
    def test_load_ml_data(self, size, directory):
        links_df, movies_df, ratings_df, tags_df = sample_data.load_ml_data(size, directory=directory)

        self.assertIsInstance(links_df, pd.DataFrame)
        self.assertIsInstance(movies_df, pd.DataFrame)
        self.assertIsInstance(ratings_df, pd.DataFrame)
        self.assertIsInstance(tags_df, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
