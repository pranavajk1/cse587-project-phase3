import re
import emoji
import pandas as pd

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