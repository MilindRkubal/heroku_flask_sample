from flask import Flask
from flask import request
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import re
from keyword_extraction import extract_phrases_keywords
import spacy
from Use_NN import use_neural_network
print('before error')
nlp = spacy.load("en_core_web_sm")
print('after error')
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")
import os
app = Flask(__name__)





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

def extract_named_entities(text):
    # text = "My birth date is 1st July 2019,But Google is starting from behind. The company made a late push into hardware, and Apple’s Siri, available on iPhones, and Amazon’s Alexa software, which runs on its Echo and Dot devices, have clear leads in consumer adoption,I hate Mumbai"

    doc = nlp(text)
    dict1 = {'ORG': 'Organisation', 'PERSON': 'Name', 'NORP': 'Nationalities group', 'FAC': 'Location',
             'GPE': 'Country/City/State', 'LOC': 'Area',
             'PRODUCT': 'Product', 'EVENT': 'Event', 'WORK_OF_ART': 'Book/Songs', 'LAW': 'Law', 'LANGUAGE': 'Language',
             'DATE': 'Date', 'TIME': 'Time',
             'PERCENT': 'Percent'}
    for ent in doc.ents:
        # print(ent.text, ent.start_char, ent.end_char, ent.label_)
        ner = ([(X.label_, X.text) for X in doc.ents])
    res_ner = []
    for item in ner:
        replacement = dict1[item[0]]
        res_ner.append((replacement, item[1]))
    res = {'Named_Entities':res_ner}
    res = json.dumps(res)
    return res

@app.route('/sentiment_analysis/')
def extract_sentiment():
    sentence = request.args.get('sentence')
    print('Input Sentence: ', sentence)
    sentiment_result = sentiment_analysis(sentence)
    return sentiment_result

@app.route('/sampletest/')
def sampletest():
    print('I AM IN SAMPLE TEST')
    sentence = request.args.get('sentence')
    print('Input Sentence in sampletest: ', sentence)
    return sentence

@app.route('/phrases_keyword_extraction/')
def extract_keywords_and_phrases():
    sentence = request.args.get('sentence')
    print('Input Sentence: ', sentence)
    phrases_keywords = extract_phrases_keywords(sentence)
    return phrases_keywords

@app.route('/named_entity_recognition/')
def extract_NER():
    sentence = request.args.get('sentence')
    print('Input Sentence: ', sentence)
    ner = extract_named_entities(sentence)
    return ner

##############################################
@app.route('/sarcasm_detection/')
def extract_sarcasm():
    sentence = request.args.get('sentence')
    print('Input Sentence: ', sentence)
    sarcasm = use_neural_network(sentence)
    return sarcasm

###############################################
def qna(sentence,question):
    result = predictor.predict(
        passage=sentence,
        question=str(question+"?")
    )
    answer=result['best_span_str']

    res = {'answer':answer}
    res = json.dumps(res)
    return res

@app.route('/question_answer/')
def extract_question_answer():
    sentence = request.args.get('sentence')
    question=request.args.get('question')
    print('Input Sentence: ', sentence)
    sarcasm = qna(sentence,question)
    return sarcasm







if __name__ == '__main__':
    app.run()



