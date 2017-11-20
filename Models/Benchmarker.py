from time import time

import numpy as np

from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

from PreProcessing.semanticExtractor import constructSemanticInput


"""
Class to do benchmark for a set of classifier with desired vectorizer
"""
class Benchmarker:
    def __init__(self, vectorizer, X_train, y_train, X_test, y_test, over_sampling=False):
        """
        init for vector models
        :param vectorizer: a vectorizer with .fit and .transform mathod implemented
        :param X_train: as name
        :param y_train: ...
        :param X_test: ...
        :param y_test: ...
        :param over_sampling: if perform over sampling
        """
        self.vectorizer = vectorizer
        self.vectorizer.fit(X_train)

        self.X_train = self.vectorizer.transform(X_train)
        self.y_train = y_train
        if over_sampling:
            self.X_train, self.y_train = SMOTE(random_state=42).fit_sample(self.X_train, self.y_train)
        self.X_test = self.vectorizer.transform(X_test)
        self.y_test = y_test


        self.grid_search_result = None
        self.best_model = (None, None, None) # (clf, score, pred)

    def benchmark(self, clf):
        """
        benchmark single classifier
        :param clf: a classifier to be benchmaked
        :return: tuple of (clf_descr: classifier description,
                    score: f1_score,
                    train_time,
                    test_time)
        """
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        #for XGBoost, need to convert sparse matrix to array for X
        clf.fit(self.X_train.toarray() if isinstance(clf, XGBClassifier) else self.X_train,
                self.y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        if isinstance(clf, XGBClassifier):
            # for XGBoost, need to convert sparse matrix to array for X
            pred = clf.predict(self.X_test.toarray())
        else:
            pred = clf.predict(self.X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.f1_score(self.y_test, pred, average='weighted')
        print("f1: %0.3f" % score)

        print("classification report:")
        print(metrics.classification_report(self.y_test, pred))

        clf_descr = str(clf).split('(')[0]

        #always store the best classifier with score and prediction result
        if self.best_model[1] == None or score > self.best_model[1]:
            self.best_model = (clf, score, pred)

        return clf_descr, score, train_time, test_time

    def hyper_grid_search(self):
        """
        Grid search with some hyper parameters
        :return:
        """
        results = []
        for clf, name in (
                (PassiveAggressiveClassifier(n_iter=50, random_state=42, average=0,
                                             class_weight='balanced'), "Passive-Aggressive"),
                # (KNeighborsClassifier(n_neighbors=10, n_jobs=8), "10NN"),
                # (KNeighborsClassifier(n_neighbors=20, n_jobs=8), "20NN"),
                # (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=8,
                #                         class_weight='balanced'), "100 Random forest"),
                # (RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=8,
                #                         class_weight='balanced'), "50 Random forest")
        ):
            print('=' * 80)
            print(name)
            results.append(self.benchmark(clf))

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            results.append(self.benchmark(LinearSVC(penalty=penalty, dual=False,
                                               tol=1e-3, random_state=42, class_weight='balanced')))

            # Train SGD model
            results.append(self.benchmark(SGDClassifier(max_iter=1000, class_weight='balanced',
                                                        tol=None, random_state=42, alpha=0.0001,
                                                        penalty=penalty)))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(self.benchmark(NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        results.append(self.benchmark(MultinomialNB(alpha=.01)))
        results.append(self.benchmark(BernoulliNB(alpha=.01)))

        print('=' * 80)
        print("LinearSVC with L1-based feature selection")

        results.append(self.benchmark(Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                            random_state=42, class_weight='balanced',
                                                            tol=1e-3))),
            ('classification', LinearSVC(penalty="l2",random_state=42, class_weight='balanced'))])))

        # print('=' * 80)
        # print("XGBoost Tree")
        # results.append(self.benchmark(XGBClassifier(max_depth=10, learning_rate=0.1,
        #                                             seed=42, nthread=8)))

        self.grid_search_result = results
        return results


    def plot(self, figure_fn=None):
        """
        plot the result
        :param figure_fn: filaname to save the plot
        :return:
        """
        indices = np.arange(len(self.grid_search_result))

        results = [[x[i] for x in self.grid_search_result] for i in range(4)]

        clf_names, score, training_time, test_time = results
        # training_time = np.array(training_time) / np.max(training_time)
        # test_time = np.array(test_time) / np.max(test_time)

        fig = plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="score", color='navy')
        # plt.barh(indices + .3, training_time, .2, label="training time",
        #          color='c')
        # plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        fig.savefig(figure_fn, transparent=True)
        plt.show()

