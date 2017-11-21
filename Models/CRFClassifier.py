import numpy as np
import pandas as pd
from sklearn import metrics
from itertools import chain
from collections import Counter

import pycrfsuite
from PreProcessing.semanticExtractor import constructSemanticInput, add_tfidf_to_feature
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

class CRFClassifier:
    def __init__(self, c1=1.0, c2=1e-3, max_iterations=50):
        """
        make a Classifier class that has similar interface of sklearn for easy benchmark
        :param c1: coefficient for L1 penalty
        :param c2: coefficient for L2 penalty
        :param max_iterations: for early stop
        """
        self.clf = None
        self.params = {
            'c1': c1,
            'c2': c2,
            'max_iterations': max_iterations,

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        }


    def fit(self, X_train, y_train):
        """
        like sklearn
        :param X_train:
        :param y_train:
        :return:
        """
        trainer = pycrfsuite.Trainer(verbose=True)

        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        trainer.set_params(self.params)

        trainer.train('crf_model.crfsuite')

    def predict(self, X_test):
        """
        like sklearn
        :param X_test:
        :return: predictions
        """
        tagger = pycrfsuite.Tagger()
        tagger.open('crf_model.crfsuite')

        y_pred = [tagger.tag(xseq) for xseq in X_test]
        return y_pred

    def learned_transitions(self, top_num=15):
        """
        print and return top (top_num) learned transitions
        :param top_num:
        :return:
        """
        def print_transitions(trans_features):
            for (label_from, label_to), weight in trans_features:
                print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

        tagger = pycrfsuite.Tagger()
        tagger.open('crf_model.crfsuite')

        info = tagger.info()

        print("Top likely transitions:")
        top = Counter(info.transitions).most_common(top_num)
        print_transitions(top)

        print("\nTop unlikely transitions:")
        bottom = Counter(info.transitions).most_common()[-top_num:]
        print_transitions(bottom)

        return top, bottom

    def state_features(self, top_num=20):
        """
        print and return top (top_num) state features
        :param top_num:
        :return:
        """
        def print_state_features(state_features):
            for (attr, label), weight in state_features:
                print("%0.6f %-6s %s" % (weight, label, attr))

        tagger = pycrfsuite.Tagger()
        tagger.open('crf_model.crfsuite')

        info = tagger.info()

        print("Top positive:")
        top = Counter(info.state_features).most_common(top_num)
        print_state_features(top)

        print("\nTop negative:")
        bottom = Counter(info.state_features).most_common()[-top_num:]
        print_state_features(bottom)

        return top, bottom


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics.
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return metrics.classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )

if __name__ == "__main__":
    X_train, y_train = constructSemanticInput("../data/input_train_tag.csv", "../data/cleaned_train.csv")
    X_test, y_test = constructSemanticInput("../data/input_test_tag.csv", "../data/cleaned_test.csv")

    #add tf-idf to features
    # tried adding TFIDF score, show no improvements, with or without stop words
    # df = pd.read_csv("../data/cleaned_train.csv")
    # doc = df['review']
    #
    # stops = set(stopwords.words("english"))
    # stops.add('nan')
    # vectorizer = TfidfVectorizer(stop_words=stops, ngram_range=(1, 1))
    # vectorizer.fit(doc)
    #
    # add_tfidf_to_feature(X_train, df, vectorizer)
    # add_tfidf_to_feature(X_test, pd.read_csv("../data/cleaned_test.csv"), vectorizer)


    crf = CRFClassifier()
    crf.fit(X_train, y_train)
    pred = crf.predict(X_test)
    df = pd.read_csv("../data/cleaned_test.csv")
    df['terms'] = y_test
    df['pred'] = pred
    df['terms'] = df['terms'].apply(lambda term: ",".join(term))
    df['pred'] = df['pred'].apply(lambda pred: ",".join(pred))
    df.to_csv("../results/crf_result.csv", encoding='utf-8', index=False)

    print(bio_classification_report(y_test, pred))

    crf.state_features()

    crf.learned_transitions()


