from flask import Flask, request, jsonify
from model import ReviewClassifier
import pandas as pd
import os
import json

app = Flask(__name__)
classifier = ReviewClassifier()

# Файл с отзывами
reviews_file = 'reviews.csv'
if not os.path.exists(reviews_file):
    pd.DataFrame(columns=['product_id', 'product_name', 'user_name', 'rating', 'text', 'prediction', 'confidence', 'reason', 'timestamp']).to_csv(reviews_file, index=False)

# Каталог товаров
products = [
    {'id': 1, 'name': 'iPhone 15 Pro', 'category': 'Электроника', 'price': '$999', 'rating': 4.8, 'image': '📱', 'description': 'Флагманский смартфон Apple с чипом A17 Pro'},
    {'id': 2, 'name': 'Samsung Galaxy S24', 'category': 'Электроника', 'price': '$899', 'rating': 4.7, 'image': '📱', 'description': 'Android-смартфон с AI функциями'},
    {'id': 3, 'name': 'MacBook Air M2', 'category': 'Ноутбуки', 'price': '$1199', 'rating': 4.9, 'image': '💻', 'description': 'Ультратонкий ноутбук на Apple Silicon'},
    {'id': 4, 'name': 'Sony WH-1000XM5', 'category': 'Аудио', 'price': '$399', 'rating': 4.8, 'image': '🎧', 'description': 'Беспроводные наушники с шумоподавлением'},
    {'id': 5, 'name': 'PlayStation 5', 'category': 'Гейминг', 'price': '$499', 'rating': 4.9, 'image': '🎮', 'description': 'Игровая консоль нового поколения'},
    {'id': 6, 'name': 'Kindle Paperwhite', 'category': 'Электроника', 'price': '$149', 'rating': 4.6, 'image': '📖', 'description': 'Электронная книга с подсветкой'},
    {'id': 7, 'name': 'Apple Watch Series 9', 'category': 'Гаджеты', 'price': '$399', 'rating': 4.7, 'image': '⌚', 'description': 'Умные часы с продвинутым здоровьем'},
    {'id': 8, 'name': 'Dyson V15 Detect', 'category': 'Техника', 'price': '$749', 'rating': 4.8, 'image': '🧹', 'description': 'Робот-пылесос с лазерным обнаружением'},
]

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Генерация карточек товаров
    products_html = ''
    for product in products:
        # Получаем отзывы для этого товара
        df = pd.read_csv(reviews_file)
        product_reviews = df[df['product_id'] == product['id']]
        review_count = len(product_reviews)
        
        products_html += f'''
        <div class="product-card">
            <div class="product-image">{product['image']}</div>
            <div class="product-info">
                <h3>{product['name']}</h3>
                <p class="category">{product['category']}</p>
                <p class="description">{product['description']}</p>
                <div class="product-footer">
                    <span class="price">{product['price']}</span>
                    <div class="rating">
                        <span class="stars">{"⭐" * int(product["rating"])}</span>
                        <span class="rating-value">{product["rating"]}</span>
                        <span class="review-count">({review_count} reviews)</span>
                    </div>
                </div>
                <a href="/product/{product["id"]}" class="view-btn">View Product</a>
            </div>
        </div>
        '''
    
    html = html.replace('<!-- PRODUCTS_PLACEHOLDER -->', products_html)
    return html

@app.route('/product/<int:product_id>')
def product_page(product_id):
    product = next((p for p in products if p['id'] == product_id), None)
    if not product:
        return "Product not found", 404
    
    with open('product.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Заменяем данные товара
    html = html.replace('{product_image}', product['image'])
    html = html.replace('{product_name}', product['name'])
    html = html.replace('{product_category}', product['category'])
    html = html.replace('{product_price}', product['price'])
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
            
            reviews_html += f'''
            <div class="review-card {sentiment_class}">
                <div class="review-header">
                    <span class="reviewer">{review['user_name']}</span>
                    <span class="review-rating">{"⭐" * int(review["rating"])}</span>
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
    
    html = html.replace('<!-- REVIEWS_PLACEHOLDER -->', reviews_html)
    return html

@app.route('/submit_review', methods=['POST'])
def submit_review():
    try:
        data = request.json
        product_id = int(data['product_id'])
        user_name = data['user_name']
        rating = int(data['rating'])
        text = data['text']
        
        # Классифицируем отзыв
        prediction, confidence, reason = classifier.predict(text)
        
        # Сохраняем отзыв
        df = pd.read_csv(reviews_file)
        new_review = pd.DataFrame({
            'product_id': [product_id],
            'product_name': [next(p['name'] for p in products if p['id'] == product_id)],
            'user_name': [user_name],
            'rating': [rating],
            'text': [text],
            'prediction': [prediction],
            'confidence': [float(confidence)],
            'reason': [reason],
            'timestamp': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')]
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
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/admin')
def admin():
    df = pd.read_csv(reviews_file)
    df = df.dropna()
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df = df.dropna(subset=['confidence'])
    
    # Статистика
    total_reviews = len(df)
    positive_count = len(df[df['prediction'] == 'positive'])
    negative_count = len(df[df['prediction'] == 'negative'])
    spam_count = len(df[df['prediction'] == 'spam'])
    
    # Отзывы по товарам
    product_stats = df.groupby('product_name').agg({
        'prediction': 'count',
        'rating': 'mean'
    }).round(2).reset_index()
    
    with open('admin.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Заменяем статистику
    html = html.replace('{total_reviews}', str(total_reviews))
    html = html.replace('{positive_count}', str(positive_count))
    html = html.replace('{negative_count}', str(negative_count))
    html = html.replace('{spam_count}', str(spam_count))
    
    # Генерация таблицы отзывов
    reviews_html = ''
    if len(df) > 0:
        for _, review in df.iterrows():
            sentiment_class = review['prediction']
            sentiment_icon = '✅' if review['prediction'] == 'positive' else '❌' if review['prediction'] == 'negative' else '⚠️'
            
            reviews_html += f'''
            <tr class="review-row {sentiment_class}">
                <td>{review['product_name']}</td>
                <td>{review['user_name']}</td>
                <td>{"⭐" * int(review["rating"])}</td>
                <td class="review-text-cell">{review['text']}</td>
                <td><span class="sentiment-badge {sentiment_class}">{sentiment_icon} {review['prediction'].upper()}</span></td>
                <td>{float(review["confidence"])*100:.1f}%</td>
                <td>{review['reason'] if review['reason'] != 'OK' else 'Valid'}</td>
                <td>{review['timestamp']}</td>
            </tr>
            '''
    
    html = html.replace('<!-- REVIEWS_TABLE_PLACEHOLDER -->', reviews_html)
    
    # Генерация статистики по товарам
    products_stats_html = ''
    for _, stat in product_stats.iterrows():
        products_stats_html += f'''
        <div class="product-stat">
            <h4>{stat['product_name']}</h4>
            <p>Total Reviews: {int(stat['prediction'])}</p>
            <p>Average Rating: {stat['rating']} ⭐</p>
        </div>
        '''
    
    html = html.replace('<!-- PRODUCTS_STATS_PLACEHOLDER -->', products_stats_html)
    return html

@app.route('/style.css')
def style():
    with open('style.css', 'r', encoding='utf-8') as f:
        return f.read(), 200, {'Content-Type': 'text/css'}

if __name__ == '__main__':
    app.run(debug=True, port=5000)