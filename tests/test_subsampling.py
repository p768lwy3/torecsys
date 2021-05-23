import unittest

import numpy as np
import pandas as pd

from torecsys.data import sub_sampling


class SubSamplingTestCase(unittest.TestCase):
    def test_df_sub_sampling(self):
        df = pd.DataFrame()
        samples = sub_sampling.sub_sampling(df, '', 'paper')
        self.assertNotEqual(samples.shape[0], 0)

    def test_ndarray_sub_sampling(self):
        arr = np.array([])
        samples = sub_sampling.sub_sampling(arr, '', 'paper')
        self.assertNotEqual(samples.shape[0], 0)


if __name__ == '__main__':
    unittest.main()
