import unittest

import pandas as pd

from PostProcessing.postProcessor import majority_element, conv2sentence

DATA_PATH = "../../data"


class TestPostProcessor(unittest.TestCase):
    def test_majority_element(self):
        self.assertEqual(majority_element(['B', 'B', 'I']), 'B')
        self.assertEqual(majority_element(['B']), 'B')
        with self.assertRaises(IndexError):
            majority_element([])


    def test_conv2sentence(self):
        #TODO: construct data for test case if the time is enough
        conv2sentence("../../results/result.csv",
                      "../../data/cleaned_test.csv", "../../results/tag_result.csv")

        df = pd.read_csv("../../results/tag_result.csv")
        self.assertEqual((df.iloc[21]['review']),
                         "speedy wifi connection and the long battery life 6 hrs ")
        self.assertEqual((df.iloc[21]['terms']), "B,I,I,B,O,O,B,I,O,O")
        pass


if __name__ == "__main__":
    unittest.main()