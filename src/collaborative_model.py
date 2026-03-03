import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import load_movies, load_ratings

def build_collaborative_model():
  movies = load_movies()
  ratings = load_ratings()

  user_movie_matrix = ratings.pivot_table(
    index = "userId",
    columns = "movieId",
    values = "rating"
  )

  user_movie_matrix = user_movie_matrix.fillna(0)

  movie_user_matrix = user_movie_matrix.T

  similarity_matrix = cosine_similarity(movie_user_matrix)

  movie_indices = pd.Series(
    range(len(movie_user_matrix.index)),
    index = movie_user_matrix.index
  )

  return movies, movie_user_matrix, similarity_matrix, movie_indices

def recommend_movies_collaborative(title, n = 5):
  movies, movie_user_matrix, similarity_matrix, movie_indices = build_collaborative_model()
  
  movie_id = movies[movies["title"] == title]["movieId"].values[0]

  if movie_id not in movie_indices:
    raise ValueError("Filme não encontrado na matriz.")

  idx = movie_indices[movie_id]

  sim_scores = list(enumerate(similarity_matrix[idx]))
  sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
  sim_scores = sim_scores[1: n + 1]

  movie_ids = [movie_user_matrix.index[i[0]] for i in sim_scores]

  recs = movies[movies["movieId"].isin(movie_ids)]["title"]

  return recs