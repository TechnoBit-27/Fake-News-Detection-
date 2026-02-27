import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords", quiet=True)

st.set_page_config(page_title="Fake News Detection")
st.write("APP STARTED")

ps = PorterStemmer()

def stemming(content):
    content = str(content)
    content = re.sub('[^a-zA-Z]', " ", content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stopwords.words("english")]
    return " ".join(content)

# 🔐 SAFE PICKLE LOADING
with open("vectorizer.pkl", "rb") as f:
    vector = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📰 Fake News Detection")

input_news = st.text_area("Enter the news article:")

if st.button("Predict"):
    if input_news.strip() == "":
        st.warning("Please enter some text")
    else:
        processed = stemming(input_news)
        vectorized_input = vector.transform([processed])
        prediction = model.predict(vectorized_input)

        if prediction[0] == 0:
            st.success("✅ The news article is REAL")
        else:
            st.error("🚨 The news article is FAKE")