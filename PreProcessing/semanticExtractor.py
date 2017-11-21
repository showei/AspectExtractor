import math
import re
import collections
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pycrfsuite
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd

DATA_PATH = "../data"
stops = set(stopwords.words("english"))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


def word2features(sent, i):
    """
    build CRF feature for i th word in sentance
    :param sent: the sentence
    :param i: index of the desired word
    :return: list of strings as features
    """
    #features for current word
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word.stem=' + stemmer.stem(word),
        #'word.lemma=' + lemmatizer.lemmatize(word),
        'word[:1]=' + word[:1],
        'word[:2]=' + word[:2],
        'word[:3]=' + word[:3],
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word[-1:]=' + word[-1:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.isstop=%s' % (word in stops),
        'word.length=%s' % len(word),
        #'word.long=%s' % (len(word) >= 5),
        'postag=' + postag,
        'postag[:2]=' + postag[:2]
    ]
    if i > 0:
        #features for previous word
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.stem=' + stemmer.stem(word1),
            #'-1:word.lemma=' + lemmatizer.lemmatize(word1),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.length=%s' % len(word1),
            '-1:word.isstop=%s' % (word1 in stops),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        # no previous, then it's beginning of sentence
        features.append('BOS')

    if i < len(sent) - 1:
        #features for next word
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.stem=' + stemmer.stem(word),
            #'+1:word.lemma=' + lemmatizer.lemmatize(word1),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.length=%s' % len(word1),
            '+1:word.isstop=%s' % (word1 in stops),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        #if no next, then it's end of sentence
        features.append('EOS')

    return features


def sent2features(sent):
    """
    build feature list for sentence
    :param sent: ...
    :return: list of CRF word features
    """
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(terms):
    """
    build labels for sentence
    :param terms: the terms for the disired sentence
    :return: 2D list of labels shape is [sentence * word]
    """
    return [label for label in terms.split(',')]


def conv2sentence(tagged_fn, cleaned_fn):
    """
    convolution format to sentence format without any post-processing
    :param tagged_fn: filename of convolution format csv with labels
    :param cleaned_fn: cleaned csv filename
    :return:
    """
    doc = pd.read_csv(cleaned_fn)['review']
    result_df = pd.read_csv(tagged_fn)
    conv_words = [conv.split()[2] for conv in result_df['review']]
    predictions = list(result_df['terms'])

    current = 0
    predicted_tags = []
    for i, sentence in doc.iteritems():
        review_words = sentence.split()
        word_count = len(review_words)
        current_conv_words = conv_words[current:current + word_count]
        current_predictions = predictions[current:current + word_count]
        current += word_count

        words2preds = dict(zip(current_conv_words, current_predictions))

        tag_list = [[] for i in range(word_count)]
        for j, word in enumerate(review_words):
            #mapping label to tag
            if words2preds[word] == 1:
                tag_list[j].append('B')
            if words2preds[word] == 2:
                tag_list[j].append('I')
            if words2preds[word] == 3:
                tag_list[j].append('BB')
            if words2preds[word] == 4:
                tag_list[j].append('EB')
            if words2preds[word] == 5:
                tag_list[j].append('EI')

        final_tags = []

        for j, tags in enumerate(tag_list):
            if len(tags) == 0:
                # add all 'O' tags
                final_tags.append('O')
            else:
                final_tags.append(tags[0])

        predicted_tags.append(",".join(final_tags))

    df = pd.DataFrame({'review': doc, 'terms': predicted_tags})

    return df


def tf(word, words):
    return words.count(word)/float(len(words))


def n_containing(word, doc):
    return sum(1 for sentence in doc if word in sentence.split())


def idf(word, doc):
    return math.log(len(doc) / float((1 + n_containing(word, doc))))


def tfidf(word, words, doc):
    """
    hand crafted tfidf without normalization.
    :param word:
    :param words:
    :param doc:
    :return:
    """
    return tf(word, words) * idf(word, doc)




def add_tfidf_to_feature(X, df, vectorizer):
    """
    add tfidf score to features, or any other score with customized vectorizer
    :param X: the 2D list of word features, shape: [sentence * word]
    :param df: Dataframe contains sentence format of review as column
    :param vectorizer: already fitted vectorizer
    :return: X
    """

    doc = list(df['review'].apply(lambda review: review.split()))
    word_names = vectorizer.get_feature_names()
    matrix = vectorizer.transform(df['review']).toarray()
    for i, sentence_feature in enumerate(X):
        for j, word_feature in enumerate(sentence_feature):
            try:
                word_feature.append("tfidf={:.3f}".format(matrix[i, word_names.index(doc[i][j])]))
            except ValueError: #if current word is not in word_names, assign it a 0
                word_feature.append("tfidf={:.3f}".format(0))
            if j > 0:
                try:
                    word_feature.append("-1:tfidf={:.3f}".format(matrix[i, word_names.index(doc[i][j - 1])]))
                except ValueError:
                    word_feature.append("-1:tfidf={:.3f}".format(0))
            if j < len(sentence_feature) - 1:
                try:
                    word_feature.append("+1:tfidf={:.3f}".format(matrix[i, word_names.index(doc[i][j + 1])]))
                except ValueError:
                    word_feature.append("+1:tfidf={:.3f}".format(0))
    return X


def add_frequent_terms_to_feature(X, cleaned_fn, top_num=100):
    """
    Add 'if the current word is in the top (top_num) frequent
    aspect terms from training data' in features
    :param X: the 2D list of word features, shape: [sentence * word]
    :param cleaned_fn: the cleaned csv filename
    :param top_num: as name
    :return:
    """
    df = pd.read_csv(cleaned_fn)
    terms_list = df['terms'].apply(lambda row: re.split("[ ,;:(]", str(row)))

    terms_list = [item for sublist in terms_list for item in sublist]
    frequent_terms = list(dict(collections.Counter(terms_list).most_common(top_num)).keys())


    doc = list(df['review'].apply(lambda review: review.split()))

    for i, sentence_feature in enumerate(X):
        for j, word_feature in enumerate(sentence_feature):
            if (not doc[i][j] in stops) and (doc[i][j] in frequent_terms):
                word_feature.append("FRQ_TERM")

    return X

def constructSemanticInput(tagged_fn, cleaned_fn):
    """
    construct semantic inputs for modelling
    :param tagged_fn: tagged csv filename
    :param cleaned_fn: cleaned csv filename
    :return: (X, y)
    """
    df = conv2sentence(tagged_fn, cleaned_fn)
    sentences = df['review'].apply(lambda review: nltk.pos_tag(review.split()))

    X = [sent2features(s) for s in sentences]
    y = [sent2labels(term) for term in df['terms']]

    #add if it is frequent aspect term to feature
    add_frequent_terms_to_feature(X, cleaned_fn)

    return X, y


if __name__ == "__main__":
    constructSemanticInput("{}/input_test_tag.csv".format(DATA_PATH),
                           "{}/cleaned_test.csv".format(DATA_PATH))
