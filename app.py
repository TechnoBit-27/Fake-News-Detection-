import streamlit as st
import joblib

# Load the trained pipeline
pipeline = joblib.load('fake_news_pipeline.pkl')

# App title
st.title("Fake News Detector")
st.write("Enter a news article below and see if it is real or fake.")

# Text input
article = st.text_area("News Article", height=200)

# Predict button
if st.button("Check"):
    if article.strip() == "":
        st.warning("Please enter a news article to check.")
    else:
        # Make prediction
        label = pipeline.predict([article])[0]
        prob = pipeline.predict_proba([article])[0]

        # Display results
        st.write(f"**Prediction:** {'Fake News' if label == 1 else 'Real News'}")
        st.write(f"**Probability:** Real: {prob[0]:.2f}, Fake: {prob[1]:.2f}")