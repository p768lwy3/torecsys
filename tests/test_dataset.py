import unittest

import torecsys as trs


class DatasetTestCase(unittest.TestCase):
    def test_ndarray_to_dataset(self, data):
        _ = trs.data.dataset.NdarrayToDataset(data)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
