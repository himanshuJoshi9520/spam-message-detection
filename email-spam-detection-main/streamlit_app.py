import streamlit as st
import pandas as pd
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from PIL import Image

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stopwords.words('english')]
    return " ".join(filtered_tokens)

# Load image
try:
    image = Image.open('me2.png.jpg')
    st.image(image, caption='EMAIL')
except FileNotFoundError:
    st.warning("Image not found!")

# Load or train model and vectorizer
@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    vectorizer_path = 'vectorizer.pkl'

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        st.info("🔄 Loading saved model and vectorizer...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            tfidf = pickle.load(f)
    else:
        st.warning("⚠️ No saved model found. Training new model...")

        try:
            df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
            df.columns = ['label', 'message']
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        except Exception as e:
            st.error(f"Could not load dataset. Make sure 'spam.csv' is in the same folder.\n\nError: {e}")
            st.stop()

        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(df['message'])
        y = df['label']

        model = MultinomialNB()
        model.fit(X, y)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(tfidf, f)

        st.success("✅ Model trained and saved!")

    return model, tfidf

# Load model
model, tfidf = load_model()

# Streamlit UI
st.title('📧 Email/SMS Spam Classifier')

input_sms = st.text_input('Enter your message')

option = st.selectbox("You got the message via:", ["Via Email", "Via SMS", "Other"])

if st.button('Click to Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Message Detected!")
        else:
            st.success("✅ This message is not spam.")
