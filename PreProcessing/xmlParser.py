import pandas as pd
from lxml import etree

DATA_PATH = "../data"

def extract_review(child):
    """
    extract review text and aspect terms from a single xml element
    :param child: element from lxml.etree.iterparse
    :return: pandas Dataframe with column [review, terms]
    """
    term_list = []
    review = ""
    for subchild in child:
        if subchild.tag == "aspectterm":
            term_list.append(subchild.get('term'))
        if subchild.tag == "text":
            review = subchild.text
    return pd.DataFrame({'review': [review], 'terms': [",".join(term_list)]})

def xml2csv(xml_fn, csv_fn=None):
    """
    parse xml to pandas Dataframe, optionally save to csv file as well
    :param xml_fn: xml full filename
    :param csv_fn: (optional) csv full filename, if not None, save result to the file
    :return: parsed Dataframe
    """
    root = etree.iterparse(xml_fn, load_dtd=True, html=True)
    all_reviews = []
    for _, child in root:
        all_reviews.append(extract_review(child))
    df = pd.concat(all_reviews, ignore_index=True)  # .to_csv("data/reviews_raw_train.csv", index=False)
    df.review = df.review.shift(-1)
    df = df[df.review != ""].dropna()
    if csv_fn:
        df.to_csv(csv_fn, index=False, encoding="utf8")

    return df

if __name__ == "__main__":
    xml2csv("{}/Laptops_Train_v2.xml".format(DATA_PATH), "{}/raw_train.csv".format(DATA_PATH))
    xml2csv("{}/Laptops_Test_Gold.xml".format(DATA_PATH), "{}/raw_test.csv".format(DATA_PATH))