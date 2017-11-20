import pandas as pd
import numpy as np
from sklearn import metrics


from Models.CRFClassifier import bio_classification_report


DATA_PATH = "../data"


def majority_element(target_list):
    """
    select a majority from a list
    :param target_list:
    :return:
    """
    idx, ctr = 0, 1

    for i in range(1, len(target_list)):
        if target_list[idx] == target_list[i]:
            ctr += 1
        else:
            ctr -= 1
            if ctr == 0:
                idx = i
                ctr = 1

    return target_list[idx]


def conv2sentence(result_fn, cleaned_fn, mapped_fn=None, for_prediction=True):
    """
    convert prediction result from convolution format to sentence format
    :param result_fn:
    :param cleaned_fn:
    :param mapped_fn:
    :return:
    """
    doc = pd.read_csv(cleaned_fn)['review']
    result_df = pd.read_csv(result_fn)
    conv_words = [conv.split()[2] for conv in result_df['review']]
    predictions = list(result_df['pred' if for_prediction else 'terms'])

    current = 0
    predicted_tags = []
    # for each sentence in the original doc
    for i, sentence in doc.iteritems():
        review_words = sentence.split()
        word_count = len(review_words)

        #get the subset of conv and prediction only contian current sentence
        current_conv_words = conv_words[current:current + word_count]
        current_predictions = predictions[current:current + word_count]
        current += word_count

        words2preds = dict(zip(current_conv_words, current_predictions))

        #each word can have more than one tag according to vectorized prediction
        tag_list = [[] for i in range(word_count)]
        for j, word in enumerate(review_words):

            if words2preds[word] == 1:
                tag_list[j].append('B')
            if words2preds[word] == 2:
                tag_list[j].append('I')
            if words2preds[word] == 3:
                if j + 1 < word_count:
                    tag_list[j + 1].append('B')
            if words2preds[word] == 4:
                if j > 1:
                    tag_list[j - 1].append('B')
            if words2preds[word] == 5:
                if j > 1:
                    tag_list[j - 1].append('I')
        final_tags = []

        for j, tags in enumerate(tag_list):
            if len(tags) == 0:
                final_tags.append('O')
            else:
                tag = majority_element(tags)
                if tag == 'I' and (j == 0 or len(tag_list[j - 1]) == 0):
                    final_tags.append('B')
                elif tag == 'B' and j > 0 and final_tags[-1] == 'B':
                    final_tags.append('I')
                else:
                    final_tags.append(tag)

        predicted_tags.append(",".join(final_tags))

    df = pd.DataFrame({'review': doc, 'terms': predicted_tags})
    if mapped_fn:
        df.to_csv(mapped_fn, index=False, encoding="utf8")

    return df

def convert2bio(df):
    def converter(term):
        tags = term.split(",")
        for i, tag in enumerate(tags):
            if tag == "BB":
                tags[i] = "O"
                if i + 1 < len(tags) and tags[i + 1] == "O":
                    tags[i + 1] = "B"
            if tag == "I" and (i == 0 or tags[i - 1] == "O"):
                tags[i] = "B"
            if tag == "EB":
                tags[i] = "O"
                if i > 0 and tags[i - 1] == "O":
                    tags[i - 1] = "B"
            if tag == "EI":
                tags[i] = "O"
                if i > 0 and tags[i - 1] == "O":
                    tags[i - 1] = "I"
        return tags


    return list(df['terms'].apply(converter)),\
           list(df['pred'].apply(converter))



if __name__ == "__main__":
    #some benchmark after post-processing
    df_pred = conv2sentence("../results/prediction_result.csv",
                  "../data/cleaned_test.csv", for_prediction=True)
    df_truth = conv2sentence("../results/prediction_result.csv",
                            "../data/cleaned_test.csv", for_prediction=False)
    y_pred = list(df_pred['terms'].apply(lambda term: term.split(",")))
    y_truth = list(df_truth['terms'].apply(lambda term: term.split(",")))
    print(bio_classification_report(y_truth, y_pred))

    # df_crf = pd.read_csv("../results/crf_result.csv")
    # y_truth, y_pred = convert2bio(df_crf)
    # print(bio_classification_report(y_truth, y_pred))