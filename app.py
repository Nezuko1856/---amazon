from flask import Flask, request, jsonify
from model import ReviewClassifier
import pandas as pd
import os
import json
from datetime import datetime

app = Flask(__name__)
classifier = ReviewClassifier()

# Файл с отзывами
reviews_file = 'reviews.csv'
if not os.path.exists(reviews_file):
    pd.DataFrame(columns=['product_id', 'product_name', 'user_name', 'rating', 'text', 'prediction', 'confidence', 'reason', 'timestamp']).to_csv(reviews_file, index=False)

# Каталог товаров
products = [
    {'id': 1, 'name': 'iPhone 15 Pro', 'category': 'Electronics', 'price': '$999', 'rating': 4.8, 'image': '📱', 'description': 'Flagship Apple smartphone with A17 Pro chip'},
    {'id': 2, 'name': 'Samsung Galaxy S24', 'category': 'Electronics', 'price': '$899', 'rating': 4.7, 'image': '📱', 'description': 'Android smartphone with AI features'},
    {'id': 3, 'name': 'MacBook Air M2', 'category': 'Laptops', 'price': '$1199', 'rating': 4.9, 'image': '💻', 'description': 'Ultra-thin laptop with Apple Silicon'},
    {'id': 4, 'name': 'Sony WH-1000XM5', 'category': 'Audio', 'price': '$399', 'rating': 4.8, 'image': '🎧', 'description': 'Wireless noise-canceling headphones'},
    {'id': 5, 'name': 'PlayStation 5', 'category': 'Gaming', 'price': '$499', 'rating': 4.9, 'image': '🎮', 'description': 'Next-generation gaming console'},
    {'id': 6, 'name': 'Kindle Paperwhite', 'category': 'Electronics', 'price': '$149', 'rating': 4.6, 'image': '📖', 'description': 'E-reader with built-in light'},
    {'id': 7, 'name': 'Apple Watch Series 9', 'category': 'Gadgets', 'price': '$399', 'rating': 4.7, 'image': '⌚', 'description': 'Smartwatch with advanced health features'},
    {'id': 8, 'name': 'Dyson V15 Detect', 'category': 'Appliances', 'price': '$749', 'rating': 4.8, 'image': '🧹', 'description': 'Vacuum cleaner with laser detection'},
]

@app.route('/')
def index():
    """Главная страница с товарами"""
    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Генерация карточек товаров
    products_html = ''
    for product in products:
        # Получаем отзывы для этого товара
        df = pd.read_csv(reviews_file)
        product_reviews = df[df['product_id'] == product['id']]
        review_count = len(product_reviews)
        
        # Создаем звёзды для рейтинга
        stars = "⭐" * int(product['rating'])
        
        products_html += f'''
        <div class="product-card" data-product-id="{product['id']}">
            <div class="product-image">{product['image']}</div>
            <div class="product-info">
                <h3>{product['name']}</h3>
                <p class="category">{product['category']}</p>
                <p class="description">{product['description']}</p>
                <div class="product-footer">
                    <span class="price">{product['price']}</span>
                    <div class="rating">
                        <span class="stars">{stars}</span>
                        <span class="rating-value">{product["rating"]}</span>
                        <span class="review-count">({review_count} reviews)</span>
                    </div>
                </div>
                <a href="/product/{product["id"]}" class="view-btn">View Product</a>
            </div>
        </div>
        '''
    
    # Находим место для вставки продуктов
    if '<div class="products-grid">' in html:
        start_idx = html.find('<div class="products-grid">') + len('<div class="products-grid">')
        end_idx = html.find('</div>', start_idx)
        html = html[:start_idx] + products_html + html[end_idx:]
    
    return html

@app.route('/product/<int:product_id>')
def product_page(product_id):
    """Страница товара с отзывами"""
    product = next((p for p in products if p['id'] == product_id), None)
    if not product:
        return "Product not found", 404
    
    with open('product.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Создаем строку со звёздами для рейтинга
    stars = "⭐" * int(product['rating'])
    
    # Заменяем данные товара
    html = html.replace('{product_image}', product['image'])
    html = html.replace('{product_name}', product['name'])
    html = html.replace('{product_category}', product['category'])
    html = html.replace('{product_price}', product['price'])
    html = html.replace('{product_stars}', stars)
    html = html.replace('{product_rating}', str(product['rating']))
    html = html.replace('{product_description}', product['description'])
    html = html.replace('{product_id}', str(product_id))
    
    # Получаем отзывы для этого товара
    df = pd.read_csv(reviews_file)
    product_reviews = df[df['product_id'] == product_id]
    
    reviews_html = ''
    if len(product_reviews) > 0:
        for _, review in product_reviews.iterrows():
            sentiment_class = review['prediction']
            sentiment_icon = '✅' if review['prediction'] == 'positive' else '❌' if review['prediction'] == 'negative' else '⚠️'
            
            # Создаем звёзды для рейтинга отзыва
            review_stars = "⭐" * int(review['rating'])
            
            reviews_html += f'''
            <div class="review-card {sentiment_class}">
                <div class="review-header">
                    <span class="reviewer">{review['user_name']}</span>
                    <span class="review-rating">{review_stars}</span>
                    <span class="sentiment-badge {sentiment_class}">
                        {sentiment_icon} {review['prediction'].upper()} ({float(review["confidence"])*100:.0f}%)
                    </span>
                </div>
                <p class="review-text">{review['text']}</p>
                <div class="review-footer">
                    <small>{review['timestamp']}</small>
                    {f'<small class="reason">⚠️ {review["reason"]}</small>' if review['reason'] != 'OK' else ''}
                </div>
            </div>
            '''
    else:
        reviews_html = '''
        <div class="no-reviews">
            <div style="font-size: 3em; margin-bottom: 20px;">📝</div>
            <h3>No reviews yet</h3>
            <p>Be the first to review this product!</p>
        </div>
        '''
    
    # Вставляем отзывы
    if '<!-- REVIEWS_PLACEHOLDER -->' in html:
        html = html.replace('<!-- REVIEWS_PLACEHOLDER -->', reviews_html)
    
    return html

@app.route('/submit_review', methods=['POST'])
def submit_review():
    """API для добавления нового отзыва - ОСНОВНОЙ API ДЛЯ МОДЕЛИ"""
    try:
        data = request.json
        product_id = int(data['product_id'])
        user_name = data['user_name']
        rating = int(data['rating'])
        text = data['text']
        
        # Проверка входных данных
        if not user_name or not text:
            return jsonify({'success': False, 'error': 'Name and review text are required'}), 400
        
        if rating < 1 or rating > 5:
            return jsonify({'success': False, 'error': 'Rating must be between 1 and 5'}), 400
        
        # Классифицируем отзыв с помощью нейронной сети
        prediction, confidence, reason = classifier.predict(text)
        
        # Получаем имя товара
        product_name = next((p['name'] for p in products if p['id'] == product_id), 'Unknown Product')
        
        # Сохраняем отзыв
        df = pd.read_csv(reviews_file)
        new_review = pd.DataFrame({
            'product_id': [product_id],
            'product_name': [product_name],
            'user_name': [user_name],
            'rating': [rating],
            'text': [text],
            'prediction': [prediction],
            'confidence': [float(confidence)],
            'reason': [reason],
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M')]
        })
        
        df = pd.concat([df, new_review], ignore_index=True)
        df.to_csv(reviews_file, index=False)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence, 3),
            'reason': reason
        })
        
    except Exception as e:
        print(f"Error submitting review: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/admin')
def admin():
    """Админ-панель с статистикой"""
    df = pd.read_csv(reviews_file)
    
    # Проверяем, есть ли данные
    if len(df) == 0:
        total_reviews = 0
        positive_count = 0
        negative_count = 0
        spam_count = 0
    else:
        df = df.dropna()
        if 'confidence' in df.columns:
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
            df = df.dropna(subset=['confidence'])
        
        # Статистика
        total_reviews = len(df)
        positive_count = len(df[df['prediction'] == 'positive']) if 'prediction' in df.columns else 0
        negative_count = len(df[df['prediction'] == 'negative']) if 'prediction' in df.columns else 0
        spam_count = len(df[df['prediction'] == 'spam']) if 'prediction' in df.columns else 0
    
    # Отзывы по товарам
    if len(df) > 0 and 'product_name' in df.columns and 'rating' in df.columns:
        product_stats = df.groupby('product_name').agg({
            'prediction': 'count',
            'rating': 'mean'
        }).round(2).reset_index()
    else:
        product_stats = pd.DataFrame(columns=['product_name', 'prediction', 'rating'])
    
    with open('admin.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Заменяем статистику
    html = html.replace('{total_reviews}', str(total_reviews))
    html = html.replace('{positive_count}', str(positive_count))
    html = html.replace('{negative_count}', str(negative_count))
    html = html.replace('{spam_count}', str(spam_count))
    
    # Получаем статистику модели
    model_stats = classifier.get_stats()
    
    # Генерация таблицы отзывов
    reviews_html = ''
    if len(df) > 0 and 'prediction' in df.columns:
        for _, review in df.iterrows():
            sentiment_class = review['prediction'] if 'prediction' in review else 'unknown'
            sentiment_icon = '✅' if sentiment_class == 'positive' else '❌' if sentiment_class == 'negative' else '⚠️'
            
            # Создаем звёзды для рейтинга
            review_stars = "⭐" * int(review['rating']) if 'rating' in review else ''
            
            # Безопасное получение значений
            text = review['text'] if 'text' in review else ''
            confidence = float(review['confidence']) if 'confidence' in review else 0
            reason = review['reason'] if 'reason' in review else ''
            timestamp = review['timestamp'] if 'timestamp' in review else ''
            user_name = review['user_name'] if 'user_name' in review else ''
            product_name = review['product_name'] if 'product_name' in review else ''
            
            reviews_html += f'''
            <tr class="review-row {sentiment_class}">
                <td>{product_name}</td>
                <td>{user_name}</td>
                <td>{review_stars}</td>
                <td class="review-text-cell">{text[:100]}{"..." if len(text) > 100 else ""}</td>
                <td><span class="sentiment-badge {sentiment_class}">{sentiment_icon} {sentiment_class.upper()}</span></td>
                <td>{confidence*100:.1f}%</td>
                <td>{reason if reason != 'OK' else 'Valid'}</td>
                <td>{timestamp}</td>
            </tr>
            '''
    else:
        reviews_html = '''
        <tr>
            <td colspan="8" style="text-align: center; padding: 40px;">
                <div style="font-size: 3em;">📝</div>
                <h3>No reviews yet</h3>
                <p>Reviews will appear here once users start submitting them</p>
            </td>
        </tr>
        '''
    
    html = html.replace('<!-- REVIEWS_TABLE_PLACEHOLDER -->', reviews_html)
    
    # Генерация статистики по товарам
    products_stats_html = ''
    if len(product_stats) > 0:
        for _, stat in product_stats.iterrows():
            products_stats_html += f'''
            <div class="product-stat">
                <h4>{stat['product_name']}</h4>
                <p>Total Reviews: {int(stat['prediction'])}</p>
                <p>Average Rating: {stat['rating']:.1f} ⭐</p>
            </div>
            '''
    else:
        products_stats_html = '''
        <div class="product-stat">
            <h4>No reviews yet</h4>
            <p>Product statistics will appear here</p>
        </div>
        '''
    
    html = html.replace('<!-- PRODUCTS_STATS_PLACEHOLDER -->', products_stats_html)
    
    return html

@app.route('/style.css')
def style():
    """CSS файл"""
    with open('style.css', 'r', encoding='utf-8') as f:
        return f.read(), 200, {'Content-Type': 'text/css'}

@app.route('/favicon.ico')
def favicon():
    """Favicon"""
    return '', 204

@app.errorhandler(404)
def not_found(error):
    """Обработка 404 ошибок"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Обработка 500 ошибок"""
    print(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':

    
    # Проверяем отзывы
    if os.path.exists(reviews_file):
        df = pd.read_csv(reviews_file)
        print(f" Всего отзывов в базе: {len(df)}")
    

    print("  • http://localhost:5000/ - Главная страница")

    
    app.run(debug=True, port=5000)
