# movie_recommendation_system.py (Corrected Version)
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
from scipy.sparse import csr_matrix
import difflib

app = Flask(__name__)
app.secret_key = 'your-secret-key'
CORS(app, supports_credentials=True)

# Simulated in-memory user DB and preferences
users = {}
user_preferences = {}  # {username: {movie_id: rating}}

# Load and preprocess data
try:
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    ratings = pd.read_csv("ratings_small.csv")
except FileNotFoundError:
    print("Error: Data files not found.")
    exit()

movies = movies[['id', 'title']].dropna()
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])
movies['id'] = movies['id'].astype(int)
movies['title_lower'] = movies['title'].str.lower()

ratings = ratings.dropna(subset=['movieId', 'userId', 'rating'])
ratings['movieId'] = ratings['movieId'].astype(int)

valid_movie_ids = set(movies['id'].unique())
ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]

# Movie title mappings
movie_id_to_title = dict(zip(movies['id'], movies['title']))
title_to_movie_id = {v.lower(): k for k, v in movie_id_to_title.items()}

def find_closest_title(title):
    """Fuzzy match and return movie ID"""
    if not title:
        return None
    title = title.strip().lower()
    if title in title_to_movie_id:
        print(f"[MATCH] Exact match for: {title}")
        return title_to_movie_id[title]
    matches = difflib.get_close_matches(title, title_to_movie_id.keys(), n=1, cutoff=0.6)
    if matches:
        print(f"[FUZZY] '{title}' matched with '{matches[0]}'")
        return title_to_movie_id[matches[0]]
    print(f"[NO MATCH] No match for: {title}")
    return None

# KNN model
movie_user_mat = ratings.pivot_table(index='movieId', columns='userId', values='rating', fill_value=0)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_user_mat.values)

# KMeans user clustering
user_movie_mat = ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
sparse_user_movie = csr_matrix(user_movie_mat.values)

n_clusters = min(5, len(user_movie_mat) - 1)
if n_clusters > 1:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    user_clusters = kmeans.fit_predict(sparse_user_movie)
    user_movie_mat['cluster'] = user_clusters

    cluster_top_movies = {}
    for cluster_id in range(n_clusters):
        cluster_users = user_movie_mat[user_movie_mat['cluster'] == cluster_id]
        cluster_ratings = ratings[ratings['userId'].isin(cluster_users.index)]
        top_movies = cluster_ratings.groupby('movieId')['rating'] \
            .mean().sort_values(ascending=False).head(10).index
        cluster_top_movies[cluster_id] = [movie_id_to_title.get(mid) for mid in top_movies if mid in movie_id_to_title]
else:
    cluster_top_movies = {0: []}

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username in users:
        return jsonify({'error': 'User already exists'}), 409

    users[username] = generate_password_hash(password)
    user_preferences[username] = {}
    return jsonify({'message': 'User registered successfully'})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username not in users or not check_password_hash(users[username], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    session['username'] = username
    return jsonify({'message': 'Login successful'})

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'message': 'Logged out'})

@app.route('/rate', methods=['POST'])
def rate_movie():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.json
    title = data.get('title')
    rating = float(data.get('rating'))

    movie_id = find_closest_title(title)
    if not movie_id:
        return jsonify({'error': 'Movie not found'}), 404

    username = session['username']
    user_preferences[username][movie_id] = rating
    return jsonify({'message': 'Rating saved'})

@app.route('/recommend/personalized', methods=['GET'])
def recommend_personalized():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    username = session['username']

    if username in user_preferences and user_preferences[username]:
        try:
            movie_indices = {mid: idx for idx, mid in enumerate(user_movie_mat.columns[:-1])}
            user_vector = np.zeros(len(movie_indices))

            for movie_id, rating in user_preferences[username].items():
                idx = movie_indices.get(movie_id)
                if idx is not None:
                    user_vector[idx] = rating

            predicted_cluster = kmeans.predict(user_vector.reshape(1, -1))[0]
            return jsonify(cluster_top_movies.get(predicted_cluster, []))
        except Exception as e:
            print(f"Error in personalized recommendation: {e}")

    top_rated = ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(10).index
    titles = [movie_id_to_title.get(mid) for mid in top_rated if mid in movie_id_to_title]
    return jsonify(titles)

@app.route('/recommend/similar', methods=['GET'])
def recommend_similar():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'No title provided'}), 400

    movie_id = find_closest_title(title)
    if not movie_id:
        return jsonify({'error': 'Movie not found'}), 404

    if movie_id not in movie_user_mat.index:
        return jsonify({'error': 'No data available for this movie'}), 404

    distances, indices = model_knn.kneighbors([movie_user_mat.loc[movie_id]], n_neighbors=6)
    recommendations = []
    for i in indices.flatten()[1:]:
        recommended_id = movie_user_mat.index[i]
        title = movie_id_to_title.get(recommended_id)
        if title and title not in recommendations:
            recommendations.append(title)

    return jsonify(recommendations[:5])

@app.route('/popular', methods=['GET'])
def popular_movies():
    top_rated = ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(10).index
    titles = [movie_id_to_title.get(mid) for mid in top_rated if mid in movie_id_to_title]
    return jsonify(titles)

if __name__ == '__main__':
    app.run(debug=True)
