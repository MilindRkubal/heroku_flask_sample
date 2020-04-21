from flask import Flask
from flask import request
from textblob import TextBlob


import json
import os

import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
app = Flask(__name__)

@app.route('/demo_test/')
def demo_test():
    return "Working"


@app.route('/')
def home():
    return "Home Page Working"

def get_positive_negative_words(ip):
    ip = re.sub(r'[^\w\s]', '', ip)
    test_subset = ip.split()
    sid = SentimentIntensityAnalyzer()
    pos_word_list = []
    neu_word_list = []
    neg_word_list = []

    for word in test_subset:
        if (sid.polarity_scores(word)['compound']) >= 0.5:
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.5:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)

    # print('Positive :', pos_word_list)
    # print('Neutral :', neu_word_list)
    # print('Negative :', neg_word_list)
    return (pos_word_list, neg_word_list, neu_word_list)


def sentiment_analysis(ip):
    print('IPPP: ', ip)

    blob = TextBlob(ip)
    sentiment_score = blob.sentiment

    print('got sentiment')

    words = get_positive_negative_words(ip)

    if (sentiment_score[0]>0.1):
        sentiment_pol = 'Positive'
    elif (sentiment_score[0]<0):
        sentiment_pol = 'Negative'
    else:
        sentiment_pol = 'Neutral'

    print('SSSSSSSSSSSSSSSSSSSS')

    temp = {'Sentiment':sentiment_pol,
                   'Weightage':str(sentiment_score[0]*100)+'%',
                   'Subjectivity':sentiment_score[1],
                   'Positive Words':words[0],
                   'Negative Words':words[1],
                   'Neutral Words':words[2]}

    #final_sentiment.update({each_sentence:temp})

    final_sentiment = json.dumps(temp)
    return final_sentiment

@app.route('/sentiment_analysis/')
def extract_sentiment():
    sentence = request.args.get('sentence')
    print('Input Sentence: ', sentence)
    sentiment_result = sentiment_analysis(sentence)
    return sentiment_result
