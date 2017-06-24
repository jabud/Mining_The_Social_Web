# -*- coding: UTF-8 -*-
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# load training dataset
trainset_df = pd.read_csv('/home/jfreek/workspace/Mining_The_Social_Web/datasets/alltrainset.csv', 
                      sep='\t', header=0, names=['category', 'text'])
# initialize model to vectorize
vec = TfidfVectorizer(lowercase=True, use_idf = True, norm=None, smooth_idf=True, 
					analyzer='word', input='content', stop_words=None)
# train
vec.fit_transform(['here', 'goes', 'the', 'data', 'text'])
# save the model
filename = 'tfidfsmooth.sav'
pickle.dump(vec, open(filename, 'wb'))