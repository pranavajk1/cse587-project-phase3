import re
import emoji
import pandas as pd
from sklearn import feature_extraction
import nltk
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

import joblib

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess(sentiment_data: pd.DataFrame) -> pd.DataFrame:
  emoji_pattern = re.compile("["
    u"U0001F600-U0001F64F"
    u"U0001F300-U0001F5FF"
    u"U0001F680-U0001F6FF"
    u"U0001F1E0-U0001F1FF"
    u"U00002702-U000027B0"
    u"U000024C2-U0001F251"
    "]+", flags=re.UNICODE
)

  sentiment_data['text'] = sentiment_data['text'].str.replace(r'[^\w\s]+', '')
  sentiment_data['text'] = sentiment_data['text'].str.replace(r'\s+', ' ')
  sentiment_data['text'] = sentiment_data['text'].str.replace(emoji_pattern, '', regex=True)

  sentiment_data['text'] = sentiment_data['text'].apply(emoji.demojize)

  return sentiment_data['text']

def preprocess_text(text):
  stop_words = set(stopwords.words('english'))
  stemmer = PorterStemmer()
  words = word_tokenize(text.lower())
  filtered_words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
  return ' '.join(filtered_words)

def preprocess_text_nn(text):
  max_words = 10000  # Maximum number of words to keep in the vocabulary
  max_len = 280  # Maximum sequence length

  tokenizer = Tokenizer(num_words=max_words)
  tokenizer.fit_on_texts(text)
  sequences = tokenizer.texts_to_sequences(text)
  padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
  return padded_sequences


ann_model = tf.keras.models.load_model('./models/ann_model.h5')
rnn_model = tf.keras.models.load_model('./models/rnn_model.h5')
lstm_model = tf.keras.models.load_model('./models/lstm_model.h5')

multinomial_nb = joblib.load('./models/multinomial_naive_bayes_model.pkl')
logistic_regression = joblib.load('./models/logistic_regression_model.pkl')
linear_svm = joblib.load('./models/linear_svm_model.pkl')

def predict_text(text, model):
  if model not in ['Multinomial Naive Bayes', 'Logistic', 'Linear SVM']:
    preprocessed_text = preprocess_text_nn(text)
    if model == 'ANN':
      prediction =  ann_model.predict(preprocessed_text)
    elif model == 'RNN':
      return rnn_model.predict(preprocessed_text)
    elif model == 'LSTM':
      return lstm_model.predict(preprocessed_text)
  else:
    preprocessed_text = preprocess_text(text)
    vectorizer = CountVectorizer()
    vectorizer.fit(preprocessed_text)
    vector = vectorizer.transform(preprocessed_text)
    if model == 'Multinomial Naive Bayes':
      prediction =  multinomial_nb.predict(vector)
    elif model == 'Logistic':
      prediction =  logistic_regression.predict(vector)
    elif model == 'Linear SVM':
      prediction =  linear_svm.predict(vector)
    return prediction