# -*- coding: UTF-8 -*-
import pandas as pd
from nltk.corpus import stopwords
import re
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


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
# Training data
X = trainset_df['text'].values
y = trainset_df['category'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=6)
# stopwords
stop_words = set(stopwords.words('english'))
# initialize model to vectorize
vec = TfidfVectorizer(lowercase=True, use_idf = True, norm=None, smooth_idf=False, 
	analyzer='word', input='content', stop_words=None, min_df=10, max_features=20000)

# SVM classifier ****************************************************************
svm_clf =svm.LinearSVC(C=0.1)
# Pipeline
vec_svm = Pipeline([('vectorize', vec), ('svm', svm_clf)])

t0 = time.time()
vec_svm.fit(x_train, y_train)
t1 = time.time()
total = t1 - t0
print "total time: " + str(total)

# get average accuracy
result = vec_svm.score(x_test, y_test)
# predict 0 or 1 Conf Matrix
y_pred = vec_svm.predict(x_test)
confusion_m = confusion_matrix(y_test, y_pred)
# show results:
print result
print confusion_m
print(classification_report(y_test, y_pred))
# train with all data
vec_svm.fit(X, y)
# save model
filename = '/home/workspace/Mining_The_Social_Web/models/svmtfidf20k.pkl'
joblib.dumb(vec_svm ,filename)

# Naive Bayes Classifier ****************************************************************
mnb_clf = naive_bayes.MultinomialNB()
# Pipeline
vec_nb = Pipeline([('vectorize', vec), ('mnb', mnb_clf)])

t0 = time.time()
vec_nb.fit(x_train, y_train)
t1 = time.time()
total = t1 - t0
print "total time: " + str(total)

# get average accuracy
result = vec_nb.score(x_test, y_test)
# predict 0 or 1 Conf Matrix
y_pred = vec_nb.predict(x_test)
confusion_m = confusion_matrix(y_test, y_pred)
# show results:
print result
print confusion_m
print(classification_report(y_test, y_pred))
