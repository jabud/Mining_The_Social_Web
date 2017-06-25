# -*- coding: UTF-8 -*-
import pandas as pd
import joblib
from nltk.corpus import stopwords
  
# load the model
filename = '/home/jfreek/workspace/Mining_The_Social_Web/models/tfidfsw.sav'
vectorizer = joblib.load(filename)
# create new vectors
vectorizer.transform(['Insert new string to transform here'])