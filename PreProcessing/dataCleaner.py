import pandas as pd
import re

from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

DATA_PATH = "../data"
#nltk.download("all")


def split_review(df):
    """
    split one row that contain multiple sub-sentences to multiple row with single sub-sentence
    :param df: Dataframe with column {review, terms}
    :return: Dataframe with column {review, terms}
    """
    def spliter(row):
        reviews = re.split("[,;:(]", row['review'])
        # = row['review'].split(",")
        s = pd.Series(row['terms'] if isinstance(row['terms'], str) else "", index=list(set(reviews)))
        return s
    df = df.apply(spliter, axis=1).stack().to_frame().reset_index(level=1, drop=False)
    df.columns = ['review', 'terms']
    df.reset_index(drop=True, inplace=True)
    df = df[(df.review != "")&(df.review != " ")]
    df['review'] = df['review'].apply(lambda review: review[1:] if review[0] == " " else review)

    return df

def match_terms(df):
    def matcher(row):
        terms = row['terms'].split(",")
        sub_terms = []
        for term in terms:
            if term in row['review']:
                sub_terms.append(term)
        return ",".join(sub_terms)
    df['terms'] = df.apply(matcher, axis=1)
    return df


def text_cleaner(text, deep_clean=False):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # this re.sub part credit to Kaggle #

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\.", "", text)
    text = re.sub(r"\?", "", text)
    text = re.sub(r"!", "", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub('- ', ', ', text)
    text = re.sub(',+', ',', text)
    text = re.sub('-+', '-', text)
    text = re.sub('\++', '+', text)
    text = re.sub(' - ', ' ', text)
    text = re.sub(' +', ' ', text)
    text = re.sub(' were | was | am | is | are ', ' be ', text)
    text = re.sub(' has | had ', ' have ', text)


    if deep_clean:
        words = text.split(" ")
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(word) for word in words]

        # lemmatizer = WordNetLemmatizer()
        # words = [lemmatizer.lemmatize(word) for word in words]
        text = " ".join(words)

    return text


def remove_extras(df):
    df['review'] = df['review'].apply(text_cleaner)
    df['terms'] = df['terms'].apply(text_cleaner)
    return df

def clean_pipeline(raw_fn, cleaned_fn):
    df = pd.read_csv(raw_fn)

    df = remove_extras(df)
    df = split_review(df)
    df = match_terms(df)

    if cleaned_fn:
        df.to_csv(cleaned_fn, index=False, encoding="utf8")

    return df


if __name__ == "__main__":
    clean_pipeline("{}/raw_train.csv".format(DATA_PATH), "{}/cleaned_train.csv".format(DATA_PATH))
    clean_pipeline("{}/raw_test.csv".format(DATA_PATH), "{}/cleaned_test.csv".format(DATA_PATH))
