import pandas as pd

from PreProcessing.dataCleaner import split_review

DATA_PATH = "../data"

def construct_classification_input(df):
    """
    for warming up notebook experiment
    :param df:
    :return:
    """
    df['terms'] = df['terms'].apply(lambda terms: 0 if not isinstance(terms, str) else 1)
    return df


def construct_convolutions(df):
    """
    construct a Dataframe with reviews as convolution format
    :param df: Dataframe, already cleaned data
    :return: Dataframe
    """
    def convoluter(review):
        words = review.split()
        length = len(words)
        convolutions = []
        for i, word in enumerate(words):
            #make sure the center of window always not empyt
            word_window = [
                words[i - 2] if i - 2 >= 0 else "nan",
                words[i - 1] if i - 1 >= 0 else "nan",
                word,
                words[i + 1] if i + 1 < length else "nan",
                words[i + 2] if i + 2 < length else "nan"
            ]
            convolutions.append(" ".join(word_window))
        return ",".join(convolutions)

    df['review'] = df['review'].apply(convoluter)
    df = split_review(df) # split again
    return df


def construct_tag_input(conv_df):
    """
    construct a Dataframe with tags for classification
    :param conv_df: convolution formatted Dataframe
    :return: convolution Dataframe with class labels.
    """
    def tagger(row):
        """
        Label one single row, labelling based on the central word of convolution window,
        used with df.apply
        :param row: one single row
        :return: labeled row
        """
        words = row['review'].split()
        tags = ['O'] * 5

        #class is 0 if no terms
        if (not isinstance(row['terms'], str)) or row['terms'] == "":
            return 0

        terms = row['terms'].split(",")

        for term in terms:
            sub_terms = term.split() #one term can contain multiple words
            for i, word in enumerate(words):
                for j, sub_term in enumerate(sub_terms):
                    #if a word match a word in a term
                    if word == sub_term:
                        #if look at the first word in a term
                        if j == 0:
                            #if a term only contain one word, the current word a B
                            if len(sub_terms) == 1:
                                tags[i] = 'B'
                            #if a term contain more than one word, next word has to match
                            elif i < len(words) - 1 and words[i + 1] == sub_terms[j + 1]:
                                tags[i] = 'B'
                        #previous word match previous word in term, current word is I
                        elif i > 0 and words[i - 1] == sub_terms[j - 1]:
                            tags[i] = 'I'

            #further tag BB (before B), EB: (after B), EI(after I)
            for i, tag in enumerate(tags):
                if tag == 'B':
                    if i > 0  and tags[i - 1] == 'O':
                        tags[i - 1] = 'BB'
                    if i < len(tags) - 1 and tags[i + 1] == 'O':
                        tags[i + 1] = 'EB'
                elif tag == 'I':
                    if i < len(tags) - 1 and tags[i + 1] == 'O':
                        tags[i + 1] = 'EI'

        #map tags to classes
        map_tag2cls = {'O': 0, 'B': 1, 'I': 2, 'BB': 3, 'EB': 4, 'EI': 5}

        return map_tag2cls[tags[2]]

    conv_df['terms'] = conv_df.apply(tagger, axis=1)
    return conv_df


def generate_input_csv(cleaned_fn, input_fn_prefix):
    """
    Generate input files for modelling
    :param cleaned_fn: cleaned csv filename
    :param input_fn_prefix: desired input filename prefix
    :return:
    """
    df = pd.read_csv(cleaned_fn)

    # cls_df = construct_classification_input(df)
    # cls_df.to_csv("{}_classification.csv".format(input_fn_prefix), index=False, encoding="utf8")

    conv_df = construct_convolutions(df)
    conv_df.to_csv("{}_conv.csv".format(input_fn_prefix), index=False, encoding="utf8")
    conv_df = construct_tag_input(conv_df)

    conv_df.to_csv("{}_tag.csv".format(input_fn_prefix), index=False, encoding="utf8")


if __name__ == "__main__":
    generate_input_csv("{}/cleaned_train.csv".format(DATA_PATH), "{}/input_train".format(DATA_PATH))
    generate_input_csv("{}/cleaned_test.csv".format(DATA_PATH), "{}/input_test".format(DATA_PATH))