# -*- coding: UTF-8 -*-
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re


def replace_text(text, replace_list, replace_by):
    """
    Replaces items in replace_list by items in replace_by, from a text.
    :param text: str
    :param replace_list: list 
    :param replace_by: str
    :return: new text: str
    """
    if replace_list:
        replace_list = list(set(replace_list))
        for i in xrange(len(replace_list)):
            text = text.replace(replace_list[i], replace_by.format(replace_list[i]))
    return text


def preprocess(tset, to_unicode=True):
    """
    Replaces undesirable characters and transform to unicode if needed. 
    :param tset: str
    :param to_unicode:bool 
    :return: clean text: str
    """
    # undesirable chars out!
    to_del = re.findall(r"[^\w\d\s+-.,!@#$%^&*();\\\/|<>:\"\']", tset, re.IGNORECASE)
    tset = replace_text(text=tset, replace_list=to_del, replace_by="")
    if to_unicode and type(tset) != unicode:
        tset = tset.decode('utf8', 'ignore')
    tset = re.sub(r"\s{2,}", " ", tset)
    return tset

# load training dataset
trainset_df = pd.read_csv('/home/jfreek/workspace/Mining_The_Social_Web/datasets/alltrainset.csv', 
                      sep='\t', header=0, names=['category', 'text'])
# preprocess text
trainset_df['text'] = trainset_df['text'].map(lambda x: preprocess(tset=x) if x else x)
# create a list of documents
doclist = trainset_df['text'].tolist()
# stopwords
stop_words = set(stopwords.words('english'))
# initialize model to vectorize
vec = TfidfVectorizer(lowercase=True, use_idf = True, norm=None, smooth_idf=False, 
	analyzer='word', input='content', stop_words=stop_words, min_df=10, max_features=5000)
# train vectoriuzer
vec.fit_transform(doclist)
# save the model
filename = '/home/jfreek/workspace/Mining_The_Social_Web/models/tfidfsw.sav'
joblib.dump(vec, filename)
