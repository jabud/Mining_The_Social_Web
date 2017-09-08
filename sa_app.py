from flask import Flask
from text_classifier import *
app = Flask(__name__)

@app.route('/')
def get_sentiment():
	# get data
	ts = TwitterScraper()
	# scrape twitter
	account = 'CocaCola'
	statuses = ts.scrape_all_tweets(query='#'+account, count=100)
	# format
	df = data_format(data=statuses, name=account)
	# clean text
	df['text'] = df['text'].map(lambda x: clean_text(tset=x) if x else x)
	# load the model
	filename = '/Mining_The_Social_Web/models/svmtfidf.sav'
	clf = joblib.load(filename)
	# classify
	df['category'] = df['text'].map(lambda x: clf.predict([x])[0] if x else -1)
	pos = df['category'].value_counts(normalize=True)[1]
	neg = df['category'].value_counts(normalize=True)[0]

	return 'positive %: ' + str(pos) + '\n' + 'negative %: ' + str(neg)
