# -*- coding: UTF-8 -*-
from training import clean_text
from scrapers.twitter_scraper import TwitterScraper
import pandas as pd
import joblib


def data_format(data, name, checkpoint=True):
	# convert results to Data Frame
	df = pd.DataFrame()
	for result in data:
		df_temp = pd.DataFrame()
		df_temp.loc[0, 'created_at'] = result["created_at"]
		df_temp.loc[0, 'screen_name'] = result["user"]["screen_name"]
		df_temp.loc[0, 'text'] = result["text"]
		df = df.append(df_temp)

	df.reset_index(drop=True, inplace=True)
	if checkpoint:
		df.to_pickle('/Mining_The_Social_Web/tmp/{acc}_tweets.p'.format(acc=name))
	return df


def main():
	# get data
	ts = TwitterScraper()
	# scrape twitter
	account = 'amdryzen'
	statuses = ts.scrape_all_tweets(query='#'+account, count=100)
	# format
	df = data_format(data=statuses, name=account)
	# clean text
	df['text'] = df['text'].map(lambda x: clean_text(tset=x) if x else x)
	# load the model
	filename = '/Mining_The_Social_Web/models/svmtfidf.sav'
	clf = joblib.load(filename)
	df['category'] = df['text'].map(lambda x: clf.predict([x])[0] if x else -1)
	pos = df['category'].value_counts(normalize=True)[1]
	neg = df['category'].value_counts(normalize=True)[0]
	print ('positive %: ' + str(pos))
	print ('negative %: ' + str(neg))


if __name__ == '__main__':
    main()
