import unittest

import torecsys.data.sampledata as sampledata


class TestSampleDataMethods(unittest.TestCase):
    def test_download_ml_data(self, size, directory):
        sampledata.download_ml_data(size, directory=directory)
        self.assertEqual(True, False)

    def test_load_ml_data(self, size, directory):
        sampledata.load_ml_data(size, directory=directory)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
