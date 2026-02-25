import streamlit as st
import pickle
import re
import os

# Get current file directory
BASE_DIR = os.path.dirname(__file__)

# Load model and vectorizer safely
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Clean text function
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Predict function
def predict_sentiment(sentence):
    sentence = clean_text(sentence)
    sentence_vector = vectorizer.transform([sentence])
    prediction = model.predict(sentence_vector)
    return prediction[0]

# UI
st.title("Sentiment Analyzer using NLP")

user_input = st.text_area("Enter a sentence")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        result = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: {result}")
    else:
        st.warning("Please enter a sentence.")
