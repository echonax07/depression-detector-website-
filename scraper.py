
import pandas as pd
import snscrape.modules.twitter as sntwitter
import itertools

def scraper(username,period):
    # place id of india is b850c1bfd38f30e0
     df = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(
            'from:{}'.format(username)).get_items(), ))
     df = df.loc[:, ['date', 'content']]
     return df