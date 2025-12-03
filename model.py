import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import nltk
import re
import os
import time
from typing import Tuple, Dict, Any

class ReviewClassifier:
    def __init__(self):
        """Инициализация классификатора отзывов"""
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.lemmatizer = None
        self.stop_words = None
        self.model_info = {}
        
        # Загружаем NLTK данные
        self._load_nltk_data()
        
        # Загружаем модель и компоненты
        self._load_model_and_components()
        
        # Статистика использования
        self.stats = {
            'total_predictions': 0,
            'positive_count': 0,
            'negative_count': 0,
            'spam_count': 0,
            'avg_processing_time': 0
        }
    
    def _load_nltk_data(self):
        """Загрузка данных NLTK"""
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def _load_model_and_components(self):
        """Загрузка модели и всех компонентов"""
        
        # Проверяем наличие файлов
        required_files = ['neural_model.keras', 'vectorizer.pkl', 'label_encoder.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            raise FileNotFoundError(f"Отсутствуют файлы модели: {missing_files}")
        
        try:
            # Загрузка модели
            self.model = keras.models.load_model('neural_model.keras')
            
            # Загрузка векторизатора
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Загрузка label encoder
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Информация о модели
            self.model_info = {
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'num_features': self.vectorizer.max_features if hasattr(self.vectorizer, 'max_features') else None,
                'classes': ['negative', 'positive']
            }
            
        except Exception as e:
            raise
    
    def is_spam(self, text: str) -> Tuple[bool, str]:
        """
        Проверка текста на спам
        """
        text = str(text).strip()
        
        # Быстрые проверки
        if len(text) < 3:
            return True, "Слишком короткий текст"
        
        if len(text) > 1000:
            return True, "Слишком длинный текст"
        
        # Подсчет символов
        letters = len(re.findall(r'[a-zA-Z]', text))
        digits = len(re.findall(r'[0-9]', text))
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        total_chars = len(text)
        
        if total_chars == 0:
            return True, "Пустой текст"
        
        # Процентные соотношения
        letter_ratio = letters / total_chars
        digit_ratio = digits / total_chars
        special_ratio = special_chars / total_chars
        
        # Правила для спама
        rules = [
            (digit_ratio > 0.3, "Слишком много цифр"),
            (special_ratio > 0.3, "Много специальных символов"),
            (letter_ratio < 0.4 and total_chars > 5, "Слишком мало букв"),
            (digits >= 5 and letters <= 3, "Случайная последовательность"),
        ]
        
        for condition, reason in rules:
            if condition:
                return True, reason
        
        # Проверка на рекламные фразы
        spam_phrases = [
            r'buy now', r'limited time', r'discount', r'offer', 
            r'click here', r'call now', r'www\.', r'http://',
            r'100% free', r'money back', r'risk free', r'\$\$\$',
            r'!!!', r'%%%'
        ]
        
        text_lower = text.lower()
        for phrase in spam_phrases:
            if re.search(phrase, text_lower):
                return True, "Обнаружена рекламная фраза"
        
        # Проверка на повторяющиеся символы
        if re.search(r'(.)\1{3,}', text):
            return True, "Повторяющиеся символы"
        
        # Проверка на CAPS LOCK
        caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
        if len(caps_words) > 2:
            return True, "Слишком много заглавных букв"
        
        return False, "OK"
    
    def preprocess_text(self, text: str) -> str:
        """
        Предобработка текста
        """
        # Удаляем HTML теги
        text = re.sub(r'<[^>]+>', '', str(text))
        
        # Оставляем только буквы и пробелы
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Лемматизация и удаление стоп-слов
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def predict(self, text: str) -> Tuple[str, float, str]:
        """
        Предсказание сентимента для текста
        Возвращает: 'positive', 'negative', или 'spam'
        """
        start_time = time.time()
        
        try:
            # Сначала проверяем на спам
            is_spam_flag, spam_reason = self.is_spam(text)
            
            if is_spam_flag:
                self.stats['spam_count'] += 1
                self.stats['total_predictions'] += 1
                processing_time = time.time() - start_time
                self._update_avg_time(processing_time)
                return 'spam', 0.95, spam_reason
            
            # Предобработка текста
            processed_text = self.preprocess_text(text)
            
            if len(processed_text.strip()) < 2:
                self.stats['spam_count'] += 1
                self.stats['total_predictions'] += 1
                processing_time = time.time() - start_time
                self._update_avg_time(processing_time)
                return 'spam', 0.9, "Текст отсутствует после обработки"
            
            # Векторизация
            text_vec = self.vectorizer.transform([processed_text])
            
            # Предсказание с помощью нейронной сети
            prediction_prob = self.model.predict(text_vec.toarray(), verbose=0)[0][0]
            
            # Определение класса: positive если > 0.5, иначе negative
            sentiment = 'positive' if prediction_prob > 0.5 else 'negative'
            
            # Уверенность
            confidence = float(prediction_prob) if sentiment == 'positive' else float(1 - prediction_prob)
            
            # Обновление статистики
            if sentiment == 'positive':
                self.stats['positive_count'] += 1
            else:
                self.stats['negative_count'] += 1
            
            self.stats['total_predictions'] += 1
            
            # Время обработки
            processing_time = time.time() - start_time
            self._update_avg_time(processing_time)
            
            return sentiment, confidence, "OK"
            
        except Exception as e:
            self.stats['total_predictions'] += 1
            processing_time = time.time() - start_time
            self._update_avg_time(processing_time)
            return 'spam', 0.8, f"Ошибка обработки: {str(e)}"
    
    def _update_avg_time(self, new_time: float):
        """Обновление среднего времени обработки"""
        self.stats['avg_processing_time'] = (
            self.stats['avg_processing_time'] * (self.stats['total_predictions'] - 1) + new_time
        ) / self.stats['total_predictions']
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики использования
        """
        stats = self.stats.copy()
        
        # Добавляем проценты
        if stats['total_predictions'] > 0:
            stats['positive_percent'] = (stats['positive_count'] / stats['total_predictions']) * 100
            stats['negative_percent'] = (stats['negative_count'] / stats['total_predictions']) * 100
            stats['spam_percent'] = (stats['spam_count'] / stats['total_predictions']) * 100
        else:
            stats['positive_percent'] = 0
            stats['negative_percent'] = 0
            stats['spam_percent'] = 0
        
        stats['model_info'] = self.model_info
        
        return stats
    
    def batch_predict(self, texts: list) -> list:
        """
        Пакетное предсказание для списка текстов
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append({
                'text': text,
                'prediction': result[0],
                'confidence': result[1],
                'reason': result[2]
            })
        return results
