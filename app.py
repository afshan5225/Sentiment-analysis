# sentiment_analysis_app.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
import streamlit as st
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier





import base64  # Import base64 to encode the image

def get_base64_image(image_file):
    """Convert an image file to a base64 string."""
    with open(image_file, "rb") as image:
        return base64.b64encode(image.read()).decode()

# Set the path for the background image
image_path = r"C:\Users\USER\Downloads\Untitled design (1).png"

# Add custom CSS for the background image and title styling
page_bg_image = f'''
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{get_base64_image(image_path)}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

h1 {{
    color: yellow;  /* Change the text color to yellow */
    font-size: 4em; /* Increase the font size */
    font-weight: bold; /* Make the font bold */
}}
</style>
'''

st.markdown(page_bg_image, unsafe_allow_html=True)

# Set a custom title
st.markdown("<h1>Sentiment Analysis App</h1>", unsafe_allow_html=True)

# Load the model and vectorizer
with open('random_forest_model.pkl', 'rb') as f:
    loaded_rf = pickle.load(f)

with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

# Function to preprocess text
def transform(text_column):
    output = []
    for i in text_column:
        new_text = re.sub('[^a-zA-Z]', ' ', str(i))
        new_text = new_text.lower()
        new_text = new_text.split()
        lemmatized_words = [Word(word).lemmatize() for word in new_text]
        output.append(' '.join(lemmatized_words))
    return output

# Function to classify sentiment
def classify_sentence(input_sentence):
    processed_sentence = transform([input_sentence])[0]
    vectorized_input = cv.transform([processed_sentence])
    prediction = loaded_rf.predict(vectorized_input)[0]
    return "Negative" if prediction == 0 else "Positive"

# Streamlit app

st.write("Enter a sentence below to analyze its sentiment:")

# Text input from user
input_sentence = st.text_area("Input Sentence:")

if st.button("Classify"):
    if input_sentence:
        result = classify_sentence(input_sentence)
        st.write(f"The sentiment of the input sentence is: **{result}**")
    else:
        st.write("Please enter a sentence.")

# Optional: Display additional information or statistics about your model
st.sidebar.title("Model Information")
st.sidebar.write("This app uses a Random Forest Classifier to predict sentiment (Positive/Negative) based on the input text.")
