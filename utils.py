# this is a utility file
# this will perform preproccsing of the raw tweet before feedimg to the classifier

import os
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log, sqrt
import numpy as np
import re
import ftfy
import pickle
import preprocessor as p
#p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED)
import pandas as pd
import snscrape.modules.twitter as sntwitter
import itertools
from datetime import date,timedelta
import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


from creds import CONSUMER_KEY,CONSUMER_SECRET,access_token,access_token_secret
import tweepy
# authorization of consumer key and consumer secret
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
# set access to user's access key and access secret
auth.set_access_token(access_token, access_token_secret)
# calling the api
api = tweepy.API(auth)


# Expand Contraction
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        # if url links then dont append to avoid news articles
        # also check tweet length, save those > 10 (length of word "depression")
        tweet = p.clean(tweet)
        if re.match("(\w+:\/\/\S+)", tweet) == None:
            # remove hashtag, @mention, emoji and image URLs
            #             tweet = re.sub('[^A-Za-z0-9]+', '', tweet)

            # fix weirdly encoded texts
            tweet = ftfy.fix_text(tweet)

            # expand contraction
            tweet = expandContractions(tweet)

            # remove punctuation
            tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

            # stop words
            stop_words = set(stopwords.words('english'))
            stop_words.remove('not')
            stop_words.remove('down')
            word_tokens = nltk.word_tokenize(tweet)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            tweet = ' '.join(filtered_sentence)

            # stemming words
            tweet = PorterStemmer().stem(tweet)

            cleaned_tweets.append(tweet)

    return cleaned_tweets


def scraper(username, count):
  # fetching the statuses
  statuses = api.user_timeline(username, tweet_mode='extended', count=25)

  json_data = [s._json for s in statuses]

  df = pd.io.json.json_normalize(json_data)
  start_date = df.iloc[0, 0]
  start_date = datetime.strftime(datetime.strptime(start_date, '%a %b %d %H:%M:%S +0000 %Y'), '%d-%m-%y')
  end_date = df.iloc[-1, 0]
  end_date = datetime.strftime(datetime.strptime(end_date, '%a %b %d %H:%M:%S +0000 %Y'), '%d-%m-%y')

  df = df.loc[:, ['full_text']]
  return df, start_date, end_date


def predict(data, tv, model):
  test_array = tv.transform(clean_tweets(data['full_text'])).toarray()
  prediction = model.predict(test_array)

  data['prediction'] = prediction
  return data
