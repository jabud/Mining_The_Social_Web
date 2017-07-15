# -*- coding: UTF-8 -*-
import pandas as pd
from nltk.corpus import stopwords, movie_reviews
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def preprocess(checkpoint=True):
	"""
	Reads, gives format and concatenate data frames into one.
	:param checkpoint: True to save data frame: bool 
	"""
	# getting nltk dataset:
	documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
	# data framing
	nltk_df = pd.DataFrame()
	for review, category in documents:
	    temp = pd.DataFrame(data={'text':review, 'category':category}, index=[0])
	    nltk_df = nltk_df.append(temp) 
	nltk_df.reset_index(drop=True, inplace=True)
	nltk_df['category'] = nltk_df['category'].map(lambda x: 0 if x=='neg' else 1)

	# getting tweets dataset from stanford:
	tweets_df = pd.read_csv('/Mining_The_Social_Web/datasets/tweetsstanford_training.csv', 
                       sep=',', header=None, names=['category', 'id', 'date', 'query', 'user', 'text'])
	tweets_df['category'] = tweets_df['category'].map(lambda x: 1 if x==4 else 0)

	# getting dataset from University of Michigan:
	umich_df = pd.read_csv('/Mining_The_Social_Web/datasets/umich_training.txt', 
                       sep="\t", header = None, names=['category', 'text'])

	# getting reviews dataset from Amazon:
	amazon_df = pd.read_csv('/Mining_The_Social_Web/datasets/amazon_cells_labelled.txt', 
	sep="\t", header = None, names=['text', 'category'])

	# getting review dataset from IMDB
	imdb_df = pd.read_csv('/Mining_The_Social_Web/datasets/imdb_labelled.txt', 
	sep="\t", header = None, names=['text', 'category'])

	# getting review dataset from Yelp
	yelp_df = pd.read_csv('/Mining_The_Social_Web/datasets/yelp_labelled.txt', 
	sep="\t", header = None, names=['text', 'category'])

	# concatenate ALL:
	trainset_df = pd.concat([nltk_df, tweets_df[['category', 'text']], umich_df, yelp_df,imdb_df, amazon_df])
	trainset_df.reset_index(drop=True, inplace=True)

	if checkpoint:
		trainset_df.to_csv(path_or_buf='/Mining_The_Social_Web/datasets/alltrainset.csv', 
                header=['category', 'text'], columns=['category', 'text'], index=None, sep='\t', mode='w')
	return trainset_df


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


def clean_text(tset, to_unicode=True):
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


def svm_classifier(X, y, sw=False, checkpoint=True):
	# stopwords
	stop_words = set(stopwords.words('english')) if sw else None
	# initialize model to vectorize
	vec = TfidfVectorizer(lowercase=True, use_idf = True, norm='l2', smooth_idf=False, analyzer='word', 
		input='content', stop_words=stop_words, min_df=10, max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
	# initialize svm model
	svm_clf =svm.LinearSVC(C=0.1)
	# Pipeline
	vec_svm = Pipeline([('vectorize', vec), ('svm', svm_clf)])
	# train with all data
	vec_svm.fit(X, y)
	# save model
	if checkpoint:
		filename = '/Mining_The_Social_Web/models/svmtfidf.sav'
		joblib.dump(vec_svm ,filename)
	# return 
	return vec_svm

def nb_classifier(X, y, sw=False, checkpoint=True):
	# stopwords
	stop_words = set(stopwords.words('english')) if sw else None
	# initialize model to vectorize
	vec = TfidfVectorizer(lowercase=True, use_idf = True, norm=None, smooth_idf=False, 
		analyzer='word', input='content', stop_words=stop_words, min_df=10, max_features=20000)
	# initialize
	mnb_clf = naive_bayes.MultinomialNB()
	# Pipeline
	vec_nb = Pipeline([('vectorize', vec), ('mnb', mnb_clf)])
	# fit model
	vec_nb.fit(X, y)
	# save model
	if checkpoint:
		filename = '/Mining_The_Social_Web/models/nbtfidf.sav'
		joblib.dump(vec_nb ,filename)
	return vec_nb


def main():
	# load training dataset
	trainset_df = pd.read_csv('/Mining_The_Social_Web/datasets/alltrainset.csv', 
	                      sep='\t', header=0, names=['category', 'text'])
	# preprocess text
	trainset_df['text'] = trainset_df['text'].map(lambda x: clean_text(tset=x) if x else x)
	# data
	X = trainset_df['text'].values
	y = trainset_df['category'].values
	# data partition
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=6)
	# fit model
	svmclf = svm_classifier(X=x_train, y=y_train, sw=False, checkpoint=True)
	# get average accuracy
	result = svmclf.score(x_test, y_test)
	# predict 0 or 1 Conf Matrix
	y_pred = svmclf.predict(x_test)
	# confusion matrix
	confusion_m = confusion_matrix(y_test, y_pred)
	# show results:
	print ("accuracy: " + str(result))
	print ("Confusion Matrix: \n" + str(confusion_m))
	print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
