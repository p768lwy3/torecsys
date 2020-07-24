import unittest

import torecsys.data.dataset as trs


class TestDatasetMethods(unittest.TestCase):
    def test_ndarray_to_dataset(self, data):
        dataset = trs.NdarrayToDataset(data)

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
