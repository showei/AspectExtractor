import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from Models.Benchmarker import Benchmarker



if __name__ == "__main__":
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8, min_df=2,
                                 ngram_range=(1, 1), stop_words=None)

    train_data = pd.read_csv("data/input_train_tag.csv")
    test_data = pd.read_csv("data/input_test_tag.csv")
    verify_data = test_data.copy()
    #only consider window of 3 words instead of 5
    train_data['review'] = train_data['review'].apply(lambda x: " ".join(x.split()[1:4]))
    test_data['review'] = test_data['review'].apply(lambda x: " ".join(x.split()[1:4]))

    bm = Benchmarker(vectorizer, train_data['review'], train_data['terms'],
                     test_data['review'], test_data['terms'])

    result = bm.hyper_grid_search()

    #save prediction result
    verify_data['pred'] = bm.best_model[2]
    verify_data.to_csv("results/prediction_result.csv")

    with open("results/report.txt", mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join([str(item) for item in result]))

    bm.plot("results/report.png")



