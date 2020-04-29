from flask import Flask
from flask import request
from textblob import TextBlob
from textblob import Word
import numpy as np
from keras.models import model_from_json

from keyword_extraction import extract_phrases_keywords
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from sklearn import preprocessing
from nltk.corpus import stopwords
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import SGDClassifier
import pandas as pd
from nltk.stem.porter import PorterStemmer
import pickle

# from keras.models import load_model
# MAX_SEQUENCE_LENGTH = 50 # Maximum number of words in a sentence
# MAX_NB_WORDS = 20000 # Vocabulary size
# EMBEDDING_DIM = 100 # Dimensions of Glove word vectors 
# # VALIDATION_SPLIT = 0.10
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
print('before error')

print('after error')

import json
import os

import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
app = Flask(__name__)

@app.route('/demo_test/')
def demo_test():
    return "Working"


# model_for_intent = load_model('model.h5')

def clicked(sentence):
    
    sample = pd.DataFrame([sentence])


    sample[0] = sample[0].apply(lambda x: re.sub("@[\w]*", '', x))
    sample[0] = sample[0].str.replace("[^a-zA-Z#]", " ")
    # print(df['tweets'].head(10))

    sample[0] = sample[0].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    # print(df['tweets'].head())

    tokenized_tweet = sample[0].apply(lambda x: x.split())
    # tokenized_tweet.head()

    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()

    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.lemmatize(i) for i in x])
    sample[0] = tokenized_tweet


    for i in tokenized_tweet:
        tokenized_tweet=' '.join(i)



    sample[0] = tokenized_tweet

    import pickle

    pickle_in = open("tfidf_vec2.pkl", "rb")
    tfidf_vec = pickle.load(pickle_in)
    X_test = tfidf_vec.transform(sample[0])

    loaded_model = pickle.load(open('racist.pkl', 'rb'))

    result = loaded_model.predict(X_test)
    if result == 0 :
        my_ans = "Non-Racist Comment"
    else :
        my_ans = "Racist Comment"
        
        
        
    temp = {'racist':my_ans}

   

    final_racist_sentiment = json.dumps(temp)
    return final_racist_sentiment

def emotion(sentence):
    tweets = pd.DataFrame([sentence])

    # Doing some preprocessing on these tweets as done before
    tweets[0] = tweets[0].str.replace('[^\w\s]',' ')
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    from textblob import Word
    tweets[0] = tweets[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    import pickle

    pickle_in = open("count_vec.pkl","rb")
    count_vect = pickle.load(pickle_in)
    tweet_count = count_vect.transform(tweets[0])


    # load the model from disk
    loaded_model = pickle.load(open('lsvm_model.pkl', 'rb'))
    result = loaded_model.predict(tweet_count)
    
    if result ==0:
        my_ans = "Happy Comment"
        
    else:
        my_ans = "Sad Comment"
        
    temp = {'emotion':my_ans}

   

    final_emotion = json.dumps(temp)
    return final_emotion

def get_key(val): 
  example_dict = {'AddToPlaylist': 0, 'BookRestaurant': 1, 'GetWeather': 2, 'RateBook': 3, 'SearchCreativeWork': 4, 'SearchScreeningEvent': 5}
  for key, value in example_dict.items():
              
        if val == value:
            return key
  return "key doesn't exist"

# def intent(text):  
#   tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
#   tokenizer.fit_on_texts(texts)
#   sequences = tokenizer.texts_to_sequences(texts)

#   word_index = tokenizer.word_index
#   print('Found %s unique tokens.' % len(word_index))
#   data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#   prediction = model_for_intent.predict(data)
#   test_predictions = prediction.argmax(axis=-1)
#   answer = get_key(test_predictions)
#   res = {'answer':answer}
#   res = json.dumps(res)
#   return res
    



def sarcasm(sentence):
    
    sample = pd.DataFrame([sentence])

    sample[0] = sample[0].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))

    ps = PorterStemmer()
    sample[0] = sample[0].apply(lambda x: x.split())
    sample[0] = sample[0].apply(lambda x : ' '.join([ps.stem(word) for word in x]))

    pickle_in = open("tfidf_sarcasm.pkl", "rb")
    tfidf_vect = pickle.load(pickle_in)
    X_test = tfidf_vect.transform(sample[0])

    loaded_model = pickle.load(open('sarcasm_model.pkl', 'rb'))

    model_predictions = loaded_model.predict(X_test)
    if model_predictions == 0:
        
        my_ans =  "Not sarcastic"

    else:
    
        my_ans =  "Sarcastic"
            
    temp = {'emotion':my_ans}



    final_emotion = json.dumps(temp)
    return final_emotion

def face(sentence):
    
    #load json and create model
    json_file = open('model_4layer_2_2_pool.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights from h5 file
    model.load_weights("model_4layer_2_2_pool.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
    
    img = image.load_img(sentence, target_size=(48, 48), grayscale=True)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    
    pred = model.predict_classes(img_tensor)
    print(pred,"prediction")
    return pred

    
    

    
    
 
    
    
def key(sentence):

    df_idf=pd.read_json("data/stackoverflow-data-idf.json",lines=True)
    df_idf['text'] = df_idf['title'] + df_idf['body']
    df_idf['text'] = df_idf['text'].apply(lambda x:pre_process(x))

    stopwords=get_stop_words("resources/stopwords.txt")
    docs=df_idf['text'].tolist()
    cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
    word_count_vector=cv.fit_transform(docs)
    cv=CountVectorizer(max_df=0.85,stop_words=stopwords,max_features=10000)
    word_count_vector=cv.fit_transform(docs)
    word_count_vector.shape

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names=cv.get_feature_names()
    tf_idf_vector=tfidf_transformer.transform(cv.transform([sentence]))
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    res = json.dumps(keywords)
    return res
    





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
    print("in racist")
    sentence = request.args.get('sentence')
    print(sentence,"sentence given")
    answer = clicked(sentence)
    return answer

@app.route('/sarcasm_analysis/')
def sarcasm_analysis():
    print("in sarcasm")
    sentence = request.args.get('sentence')
    print(sentence,"sentence given")
    answer = sarcasm(sentence)
    return answer


@app.route('/emotion_analysis/')
def emotion_analysis():
    print("in emotion")
    sentence = request.args.get('sentence')
    print(sentence,"sentence given")
    answer = emotion(sentence)
    return answer

@app.route('/face_emotion/')
def face_emotion():
    print("in face")
    sentence = request.args.get('sentence')
    print(sentence,"sentence given")
    answer = face(sentence)
    return answer

# @app.route('/intent_analysis/')
# def intent_analysis():
#     print("in sarcasm")
#     sentence = request.args.get('sentence')
#     print(sentence,"sentence given")
#     answer = intent(sentence)
#     return answer



@app.route('/keywords/')
def keywords():
    print("in emotion")
    sentence = request.args.get('sentence')
    print(sentence,"sentence given")
    answer = key(sentence)
    return answer
