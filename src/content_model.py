import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from src.data_loader import load_movies

movies = load_movies()

movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex = False)

vectorizer = CountVectorizer()

genre_matrix = vectorizer.fit_transform(movies["genres_clean"])

similarity_matrix = cosine_similarity(genre_matrix)

movie_indices = pd.Series(movies.index, index = movies["title"]).drop_duplicates()

def recommend_movies(title, n = 5):
  idx = movie_indices[title]

  sim_scores = list(enumerate(similarity_matrix[idx]))
  sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
  sim_scores = sim_scores[1: n + 1]

  movie_ids = [i[0] for i in sim_scores]

  return movies["title"].iloc[movie_ids]