import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
import numpy as np

# Загрузка NLTK данных
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


try:
    # Загрузка датасета с отзывами Amazon
    path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
    df = pd.read_csv(path + '/Reviews.csv')

    
except Exception as e:
    print(f"Ошибка загрузки датасета: {e}")
    print("Для работы программы необходим датасет с Kaggle")
    exit(1)

# Ограничиваем до 50к записей для быстрого обучения
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)



# Подготовка данных - используем Text и Score
df = df[['Text', 'Score']].dropna()
df['Sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else 'negative')

# Балансировка классов
positive_count = len(df[df['Sentiment'] == 'positive'])
negative_count = len(df[df['Sentiment'] == 'negative'])


# Предобработка текста
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Предобработка текста для нейронной сети"""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

print("Предобработка текста...")
df['Processed_Text'] = df['Text'].apply(preprocess_text)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    df['Processed_Text'], df['Sentiment'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['Sentiment']
)

print(f"Тренировочная выборка: {len(X_train)} записей")
print(f"Тестовая выборка: {len(X_test)} записей")

# Векторизация текста
vectorizer = TfidfVectorizer(
    max_features=2000,
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Размерность векторов: {X_train_vec.shape[1]} features")

# Кодирование меток
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# Создание нейронной сети
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_vec.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
]

print("Обучение модели...")
history = model.fit(
    X_train_vec.toarray(), y_train_enc,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_vec.toarray(), y_test_enc),
    callbacks=callbacks,
    verbose=1
)

# Оценка модели
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
    X_test_vec.toarray(), y_test_enc, verbose=0
)

y_pred = (model.predict(X_test_vec.toarray()) > 0.5).astype(int)



# Сохранение модели и компонентов

model.save('neural_model.keras')

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

