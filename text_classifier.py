# -*- coding: UTF-8 -*-
import pandas as pd
import pickle
from nltk.corpus import stopwords
  
# load the model from disk
vec_model = pickle.load(open(filename, 'rb'))
# create new vectors
vec.transform(['Insert new string to transform here'])