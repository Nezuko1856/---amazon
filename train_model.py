import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
df = pd.read_csv(path + '/Reviews.csv')

df = df[['Text', 'Score']].dropna()
df['Sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else 'negative')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['Processed_Text'] = df['Text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['Processed_Text'], df['Sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_vec.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_vec.toarray(), y_train_enc,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_vec.toarray(), y_test_enc),
    verbose=1
)

model.save('neural_model.h5')

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model training complete!")
print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print("Files saved: neural_model.h5, vectorizer.pkl, label_encoder.pkl")