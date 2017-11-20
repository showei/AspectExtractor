import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from PreProcessing.semanticExtractor import constructSemanticInput, add_tfidf_to_feature
from PreProcessing.xmlParser import xml2csv
from PreProcessing.dataCleaner import clean_pipeline
from PreProcessing.inputConstructor import generate_input_csv
from Models.Benchmarker import Benchmarker
from Models.CRFClassifier import CRFClassifier, bio_classification_report

DATA_PATH = "data"


if __name__ == "__main__":
    #download nltk data, comment it if already have it.
    print("downlaoding nltk data")
    nltk.download("all")

    #prepare csv data
    print("parsing csv data...")
    xml2csv("{}/Laptops_Train_v2.xml".format(DATA_PATH), "{}/raw_train.csv".format(DATA_PATH))
    xml2csv("{}/Laptops_Test_Gold.xml".format(DATA_PATH), "{}/raw_test.csv".format(DATA_PATH))

    #clean data
    print("cleaning data...")
    clean_pipeline("{}/raw_train.csv".format(DATA_PATH), "{}/cleaned_train.csv".format(DATA_PATH))
    clean_pipeline("{}/raw_test.csv".format(DATA_PATH), "{}/cleaned_test.csv".format(DATA_PATH))

    #format and tag data
    print("formatting and tagging data")
    generate_input_csv("{}/cleaned_train.csv".format(DATA_PATH), "{}/input_train".format(DATA_PATH))
    generate_input_csv("{}/cleaned_test.csv".format(DATA_PATH), "{}/input_test".format(DATA_PATH))

    print("vectorizing data")
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8, min_df=2,
                                 ngram_range=(1, 1), stop_words=None)

    train_data = pd.read_csv("data/input_train_tag.csv")
    test_data = pd.read_csv("data/input_test_tag.csv")
    verify_data = test_data.copy()
    #only consider window of 3 words instead of 5
    train_data['review'] = train_data['review'].apply(lambda x: " ".join(x.split()[1:4]))
    test_data['review'] = test_data['review'].apply(lambda x: " ".join(x.split()[1:4]))

    print("start to benchmark vector models")
    bm = Benchmarker(vectorizer, train_data['review'], train_data['terms'],
                     test_data['review'], test_data['terms'])

    result = bm.hyper_grid_search()

    #save prediction result
    verify_data['pred'] = bm.best_model[2]
    verify_data.to_csv("results/prediction_result.csv")

    with open("results/report.txt", mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join([str(item) for item in result]))


    #try semantic features
    print("start to benchmark semantic model")
    X_train, y_train = constructSemanticInput("data/input_train_tag.csv", "data/cleaned_train.csv")
    X_test, y_test = constructSemanticInput("data/input_test_tag.csv", "data/cleaned_test.csv")

    # add tf-idf to features
    df = pd.read_csv("data/cleaned_train.csv")
    doc = df['review']
    vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 1))
    vectorizer.fit(doc)

    # add tfidf score to feature, show no improvement
    # add_tfidf_to_feature(X_train, df, vectorizer)
    # add_tfidf_to_feature(X_test, pd.read_csv("../data/cleaned_test.csv"), vectorizer)


    crf = CRFClassifier()
    crf.fit(X_train, y_train)
    pred = crf.predict(X_test)
    df = pd.read_csv("data/cleaned_test.csv")
    df['terms'] = y_test
    df['pred'] = pred
    df['terms'] = df['terms'].apply(lambda term: ",".join(term))
    df['pred'] = df['pred'].apply(lambda pred: ",".join(pred))
    df.to_csv("results/crf_result.csv", encoding='utf-8', index=False)

    print(bio_classification_report(y_test, pred))

    crf.state_features()



