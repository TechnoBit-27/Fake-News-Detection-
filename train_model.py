import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import pickle

nltk.download('stopwords')

news_df = pd.read_csv("train.csv", nrows=5000)
vector = TfidfVectorizer(max_features=5000)
news_df = news_df.fillna("")
news_df['content'] = news_df['author'] + " " + news_df['text']

ps = PorterStemmer()

def stemming(content):
    content = str(content)
    content = re.sub('[^a-zA-Z]', " ", content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stopwords.words("english")]
    return " ".join(content)

news_df['content'] = news_df['content'].apply(stemming)

X = news_df['content'].values
y = news_df['label'].values

vector = TfidfVectorizer()
X = vector.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# SAVE MODEL
pickle.dump(vector, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model and vectorizer saved")