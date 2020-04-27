from flask import Flask
from flask import request
from textblob import TextBlob
from keyword_extraction import extract_phrases_keywords
import spacy
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

print('before error')
import json
import os

import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
app = Flask(__name__)

@app.route('/demo_test/')
def demo_test():
    return "Working"
def clicked(sentence):
    print("MY SENtence FOR ", sentence)
    train = pd.read_csv(r"train_tweets.csv", encoding='latin-1')
    test = pd.read_csv(r"test_tweets.csv", encoding='latin-1')

    df = train.append(test, ignore_index=True)

    df['tweets'] = df['tweet'].apply(lambda x: re.sub("@[\w]*", '', x))

    df['tweets'] = df['tweets'].str.replace("[^a-zA-Z#]", " ")
    # print(df['tweets'].head(10))

    df['tweets'] = df['tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    # print(df['tweets'].head())

    tokenized_tweet = df['tweets'].apply(lambda x: x.split())
    # tokenized_tweet.head()

    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()

    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.lemmatize(i) for i in x])
    # print(tokenized_tweet.head())

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    df['tweets'] = tokenized_tweet

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(df['tweets'])

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    from sklearn.naive_bayes import GaussianNB
    train_tfidf = tfidf[:31962, :]
    test_tfidf = tfidf[31962:, :]

    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_tfidf, train['label'], random_state=42,
                                                              test_size=0.3)
    xtrain_tfidf = train_tfidf[ytrain.index]
    xvalid_tfidf = train_tfidf[yvalid.index]

    with open(r'astral_submission.csv', 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(['id','tweet'])
        thewriter.writerow([1,sentence])

    sample = pd.read_csv(r"astral_submission.csv", index_col=0)


    sample['tweets'] = sample['tweet'].apply(lambda x: re.sub("@[\w]*", '', x))
    sample['tweets'] = sample['tweets'].str.replace("[^a-zA-Z#]", " ")
    # print(df['tweets'].head(10))

    sample['tweets'] = sample['tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    # print(df['tweets'].head())

    tokenized_tweet = sample['tweets'].apply(lambda x: x.split())
    # tokenized_tweet.head()

    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()

    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.lemmatize(i) for i in x])
    sample['tweets'] = tokenized_tweet


    for i in tokenized_tweet:
        tokenized_tweet=' '.join(i)



    sample['tweets'] = tokenized_tweet

    lreg= LogisticRegression()
    lreg.fit(xtrain_tfidf, ytrain)

    X_test = tfidf_vectorizer.transform(sample['tweets'])

    b =lreg.predict(X_test)
    if b == 0 :
        print("Non-Racist Comment")
        my_ans= "Non-Racist Comment"
    else :
        print("Racist Comment")
        my_ans= "Racist Comment"
        
        
        
    temp = {'racist':my_ans}

   

    final_racist_sentiment = json.dumps(temp)
    return final_racist_sentiment

@app.route('/')
def home():
    return "Home Page Working"

def qna(sentence,question):
    result = predictor.predict(
        passage=sentence,
        question=str(question+"?")
    )
    answer=result['best_span_str']

    res = {'answer':answer}
    res = json.dumps(res)
    return res

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


@app.route('/named_entity_recognition/')
def extract_NER():
    sentence = request.args.get('sentence')
    print('Input Sentence: ', sentence)
    ner = extract_named_entities(sentence)
    return ner

@app.route('/phrases_keyword_extraction/')
def extract_keywords_and_phrases():
    sentence = request.args.get('sentence')
    print('Input Sentence: ', sentence)
    phrases_keywords = extract_phrases_keywords(sentence)
    return phrases_keywords

@app.route('/sentiment_analysis/')
def extract_sentiment():
    sentence = request.args.get('sentence')
    print('Input Sentence: ', sentence)
    sentiment_result = sentiment_analysis(sentence)
    return sentiment_result

@app.route('/racist_analysis/')
def racist_analysis():
    sentence = request.args.get('sentence')
    answer = clicked(sentence)
    return answer

