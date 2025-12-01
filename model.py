import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
import re
import os

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class ReviewClassifier:
    def __init__(self):
        if not os.path.exists('neural_model.h5'):
            raise FileNotFoundError("neural_model.h5 not found. Download from link in README.md")
        
        self.model = load_model('neural_model.h5')
        
        if os.path.exists('vectorizer.pkl'):
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            raise FileNotFoundError("vectorizer.pkl not found")
        
        if os.path.exists('label_encoder.pkl'):
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
        else:
            raise FileNotFoundError("label_encoder.pkl not found")
        
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
    
    def is_spam(self, text):
        text = str(text).strip()
        
        if len(text) < 2:
            return True, "Too short"
        
        letters = len(re.findall(r'[a-zA-Z]', text))
        digits = len(re.findall(r'[0-9]', text))
        spaces = len(re.findall(r'\s', text))
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        total_chars = len(text)
        
        letter_ratio = letters / total_chars if total_chars > 0 else 0
        digit_ratio = digits / total_chars if total_chars > 0 else 0
        special_ratio = special_chars / total_chars if total_chars > 0 else 0
        
        if digit_ratio > 0.3:
            return True, "Too many digits"
        
        if special_ratio > 0.3:
            return True, "Too many special characters"
        
        if letter_ratio < 0.4 and total_chars > 5:
            return True, "Too few letters"
        
        words = text.split()
        mixed_word_count = 0
        for word in words:
            if re.search(r'[a-zA-Z]', word) and re.search(r'[0-9]', word):
                mixed_word_count += 1
        
        if mixed_word_count >= 2:
            return True, "Mixed alphanumeric words"
        
        non_english = len(re.findall(r'[^a-zA-Z0-9\s\.,!?\-]', text))
        if non_english / total_chars > 0.2:
            return True, "Non-English characters"
        
        if re.search(r'(.)\1{3,}', text):
            return True, "Repeating characters"
        
        if digits >= 3 and letters <= 2:
            return True, "Random sequence"
        
        vowels = len(re.findall(r'[aeiouAEIOU]', text))
        if letters > 5 and vowels / letters < 0.1:
            return True, "Too few vowels"
        
        return False, "OK"
    
    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text))
        text = text.lower()
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def predict(self, text):
        is_spam_flag, spam_reason = self.is_spam(text)
        
        if is_spam_flag:
            return 'spam', 0.95, spam_reason
        
        try:
            processed_text = self.preprocess_text(text)
            
            if len(processed_text.strip()) == 0:
                return 'spam', 0.9, "No text after processing"
            
            text_vec = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(text_vec.toarray(), verbose=0)[0][0]
            sentiment = 'positive' if prediction > 0.5 else 'negative'
            confidence = float(prediction) if sentiment == 'positive' else float(1 - prediction)
            return sentiment, confidence, "OK"
        except Exception as e:
            return 'spam', 0.8, f"Processing error: {str(e)}"