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


ann_model = tf.keras.models.load_model('./models/ann_model.h5')
rnn_model = tf.keras.models.load_model('./models/rnn_model.h5')
lstm_model = tf.keras.models.load_model('./models/lstm_model.h5')

multinomial_nb = joblib.load('./models/multinomial_naive_bayes_model.pkl')
logistic_regression = joblib.load('./models/logistic_regression_model.pkl')
linear_svm = joblib.load('./models/linear_svm_model.pkl')

def preprocess_text_nn(test_comments):
  max_words = 10000  # Maximum number of words to keep in the vocabulary
  max_len = 280  # Maximum sequence length
  tokenizer = Tokenizer(num_words=max_words)
  test_sequence = tokenizer.texts_to_sequences([test_comments])
  padded_test_sequence = pad_sequences(test_sequence, maxlen=max_len, padding='post', truncating='post')
  return padded_test_sequence

def nn_model_final_output(text, predictions):
  sentiment_mapping = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
  }
  predicted_label_index = predictions.argmax()
  predicted_label = list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(predicted_label_index)]
  print(f"\nComment: {text}")
  print(f"Predicted sentiment: {predicted_label}")
  return predicted_label


def preprocess_text(text):
  # Preprocessing: Convert text to lowercase, remove stopwords, and perform stemming
  stop_words = set(stopwords.words('english'))
  stemmer = PorterStemmer()

  def preprocess_text(text):
      words = word_tokenize(text.lower())
      filtered_words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
      return ' '.join(filtered_words)
  
  # Load the CountVectorizer used during training
  vectorizer_loaded = joblib.load('models/vectorizer.pkl')
  user_input_processed = preprocess_text(text)
  user_input_vectorized = vectorizer_loaded.transform([user_input_processed])
  return(user_input_vectorized)[0]

def predict_text(text, model):
  if model not in ['Multinomial Naive Bayes', 'Logistic', 'Linear SVM']:
    preprocessed_text = preprocess_text_nn(text)
    if model == 'ANN':
      predictions =  ann_model.predict(preprocessed_text)[0]
    elif model == 'RNN':
      predictions = rnn_model.predict(preprocessed_text)[0]
    elif model == 'LSTM':
      predictions =  lstm_model.predict(preprocessed_text)[0]
    final_output = nn_model_final_output(text, predictions)

  else:
    preprocessed_text = preprocess_text(text)
    if model == 'Multinomial Naive Bayes':
      prediction = multinomial_nb.predict(preprocessed_text)
    elif model == 'Logistic':
      prediction =  logistic_regression.predict(preprocessed_text)
    elif model == 'Linear SVM':
      prediction =  linear_svm.predict(preprocessed_text)
    # Convert predicted label index to a string label
    final_output = 'Neutral' if prediction == 1 else ('Negative' if prediction == 0 else 'Positive')
    print(f"\nComment: {text}")
    print(f"Predicted sentiment: {final_output}")

  return final_output