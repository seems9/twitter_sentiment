
#Read the tweets one by one and process it
import re
import csv
#pretty print in data 
import pprint
import nltk
import re
import numpy
import nltk.classify
import re, pickle, csv, os
from collections import defaultdict
import nltk.classify.util

from numpy import *



def replaceTwoOrMore(s):
  #look for 2 or more repetitions of character
  pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
  #print pattern.sub(r"\1\1", s)
  return pattern.sub(r"\1\1", s)
  #end


def processTweet(tweet):
  # process the tweets
  #Convert to lower case
  tweet = tweet.lower()
  #Convert www.* or https?://* to URL
  tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
  #Convert @username to AT_USER
  tweet = re.sub('@[^\s]+','AT_USER',tweet)
  #Remove additional white spaces
  tweet = re.sub('[\s]+', ' ', tweet)
  #Replace #word with word
  tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
 
  #trim
  tweet = tweet.strip('\'"')
  #print tweet
  return tweet
  #end
  #start getStopWordList
def getStopWordList(stopWordListFileName):
  #read the stopwords
  stopWords = []
  #stopWords.append('AT_USER')
  #stopWords.append('URL')
  fp = open(stopWordListFileName, 'r')
  line = fp.readline()
  while line:
    word = line.strip()
    stopWords.append(word)
    line = fp.readline()
  fp.close()
  return stopWords
  #end

stopWords = getStopWordList('stopwords.txt')

  #start getfeatureVector
def getFeatureVector(tweet, stopWords):
  featureVector = []
  words = tweet.split()
  for w in words:
  #replace two or more with two occurrences
    w = replaceTwoOrMore(w)
    #strip punctuation
    w = w.strip('\'"?,.')
    #check if it consists of only words
    val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
    #ignore if it is a stopWord
    if(w in stopWords or val is None):
      continue
    else:
      featureVector.append(w.lower())
  #print featureVector
  return featureVector
#end



#get feature list stored in a file (for reuse)
#featureList = getFeatureList('project/terrorism6.txt')

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)

    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)

    return features
#end



"""
inpTweets = csv.reader(open('terrorism6.csv', 'rU'), delimiter=',', quotechar='|')
tweets = []
for row in inpTweets:
  sentiment = row[0]
  #The actual tweet
  tweet = row[1]
  processedTweet = processTweet(tweet)
  featureVector = getFeatureVector(processedTweet, stopWords)
  tweets.append((featureVector, sentiment));
#end loop 
"""



#Read the tweets one by one and process it
inpTweets = csv.reader(open('labelledbadwords.csv', 'rU'), delimiter=',', quotechar='|')
stopWords = getStopWordList('stopwords.txt')
featureList = []

# Get tweet words
tweets = []
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    #print processedTweet
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));


inpTweets = csv.reader(open('terrorism6.csv', 'rU'), delimiter=',', quotechar='|')
stopWords = getStopWordList('stopwords.txt')
featureList = []

# Get tweet words
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    #print processedTweet
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));

"""
badwords = csv.reader(open('badwords.csv', 'rU'), delimiter=',', quotechar='|')

for bw in badwords:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    #print processedTweet
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));


#end loop
"""

# Remove featureList duplicates
featureList = list(set(featureList))

# Extract feature vector for all tweets in one shote
training_set = nltk.classify.util.apply_features(extract_features, tweets)



# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
# Test the classifier
inp= csv.reader(open('project/test.csv', 'rU'), delimiter=',', quotechar='|')
for test in inp:
  testTweet = test
  #testTweet = 'Congrats very good job'
  processedTestTweet = processTweet(testTweet[0])
  sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
  #acc = accuracy(classifier, [(features(n), g) for (n, g) in test])
  #print('Accuracy: %6.4f' % acc)
  print "testTweet = %s, sentiment = %s\n" % (testTweet, sentiment)








 

