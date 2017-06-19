import twitter
import os
import pandas as pd
from urllib import unquote

# TODO: ADD parameters to specify account from command line
# TODO: README, add all libraries and tools needed, if possible dockerized everything!


class TwitterScraper:
    """
    Twitter class, initializes API.
    """
    def __init__(self):
        config = {}
        execfile("config.py", config)
        self.auth = twitter.oauth.OAuth(config['TWITTER_ACCESS_KEY'], config['TWITTER_ACCESS_SECRET'],
                                        config['TWITTER_CONSUMER_KEY'], config['TWITTER_CONSUMER_SECRET'])
        self.twitter_api = twitter.Twitter(auth=self.auth)

    def scrape_n_tweets(self, query, count=None):
        """
        Scrapes n number of tweets.
        :param query: account to scrape: str
        :param count: number of tweets: int
        :return: statuses with results: list
        """
        search_results = self.twitter_api.search.tweets(q=query, count=count)
        statuses = search_results['statuses']
        return statuses

    def scrape_all_tweets(self, query, count):
        """
        Scrapes all tweets using a cursor.
        :param query: account to scrape: str
        :param count: number of tweets per page: int
        :return: statuses with results: list
        """
        search_results = self.twitter_api.search.tweets(q=query, count=count)
        statuses = search_results['statuses']
        while True:
            print "Length of statuses", len(statuses)
            try:
                next_results = search_results['search_metadata']['next_results']
            except KeyError:  # No more results when next_results doesn't exist
                break

            # Create a dictionary from next_results, which has the following form:
            # ?max_id=313519052523986943&q=NCAA&include_entities=1
            kwargs = dict([kv.split('=') for kv in unquote(next_results[1:]).split("&")])

            search_results = self.twitter_api.search.tweets(**kwargs)
            statuses += search_results['statuses']
        return statuses


def main():
    # initialize twitter class
    ts = TwitterScraper()
    # scrape twitter
    account = 'amdryzen'
    statuses = ts.scrape_all_tweets(query='#'+account, count=100)
    # convert results to Data Frame
    df = pd.DataFrame()
    for result in statuses:
        df_temp = pd.DataFrame()
        df_temp.loc[0, 'created_at'] = result["created_at"]
        df_temp.loc[0, 'screen_name'] = result["user"]["screen_name"]
        df_temp.loc[0, 'text'] = result["text"]
        df = df.append(df_temp)

    df.reset_index(drop=True, inplace=True)
    df.to_pickle('/home/jfreek/workspace/Mining_The_Social_Web/tmp/{acc}_tweets.p'.format(acc=account))

if __name__ == "__main__":
    main()
