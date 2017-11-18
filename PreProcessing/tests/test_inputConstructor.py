import pandas as pd

import unittest

from PreProcessing.inputConstructor import construct_convolutions, construct_tag_input


DATA_PATH = "../../data"


class TestInputConstructor(unittest.TestCase):
    def test_construct_convolutions(self):
        # TODO: construct data for test case if the time is enough
        df = pd.read_csv("{}\cleaned_test.csv".format(DATA_PATH))
        df = df.iloc[21:26]
        true_row_num = 0
        for i, row in df.iterrows():
            true_row_num += len(row['review'].split())

        conv_df = construct_convolutions(df)
        self.assertEqual(true_row_num,len(conv_df))
        pass

    def test_construct_tag_input(self):
        # TODO: construct data for test case if the time is enough
        df = pd.read_csv("{}\cleaned_test.csv".format(DATA_PATH))
        df = df.iloc[21:23]
        true_labels = [3, 5, 0, 0, 2, 3, 1, 2, 1, 5, 0, 2, 0, 0, 0, 0, 1, 3]
        conv_df = construct_convolutions(df)
        conv_df = construct_tag_input(conv_df)
        self.assertEqual(list(conv_df['terms']), true_labels)

if __name__ == "__main__":
    unittest.main()