import twitter
import os
import pandas as pd
from urllib import unquote


class TwitterScraper:
    def __init__(self):
        config = {}
        execfile("C:\Users\workspace\Mining_The_Social_Web\config.py", config)
        self.auth = twitter.oauth.OAuth(config['TWITTER_ACCESS_KEY'], config['TWITTER_ACCESS_SECRET'],
                                        config['TWITTER_CONSUMER_KEY'], config['TWITTER_CONSUMER_SECRET'])
        self.twitter_api = twitter.Twitter(auth=self.auth)

    def scrape_tweets(self, query, count=None):
        search_results = self.twitter_api.search.tweets(q=query, count=count)
        statuses = search_results['statuses']
        return statuses


def main():
    ts = TwitterScraper()
    statuses = ts.scrape_tweets(query='#amdryzen', count=100)
    df = pd.DataFrame()
    for result in statuses:
        df_temp = pd.DataFrame()
        df_temp.loc[0, 'created_at'] = result["created_at"]
        df_temp.loc[0, 'screen_name'] = result["user"]["screen_name"]
        df_temp.loc[0, 'text'] = result["text"]
        df = df.append(df_temp)

    df.reset_index(drop=True, inplace=True)
    df.to_pickle('~/workspace/tmp/amdryzen_tweets.p')
if __name__ == "__main__":
    main()
