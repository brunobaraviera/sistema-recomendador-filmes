import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import load_movies, load_ratings

movies = load_movies()
ratings = load_ratings()

movie_ids = ratings["movieId"].unique()

movie_map = {id: i for i, id in enumerate(movie_ids)}
reverse_movie_map = {v: k for k, v in movie_map.items()}

class RecommenderNet(nn.Module):
  def __init__(self, num_users, num_movies, embedding_size = 50):
    super().__init__()

    self.user_embedding = nn.Embedding(num_users, embedding_size)
    self.movie_embedding = nn.Embedding(num_movies, embedding_size)

    self.fc1 = nn.Linear(embedding_size * 2, 128)
    self.fc2 = nn.Linear(128, 1)

    self.relu = nn.ReLU()

  def forward(self, user, movie):
    user_emb = self.user_embedding(user)
    movie_emb = self.movie_embedding(movie)

    x = torch.cat([user_emb, movie_emb], dim = 1)

    x = self.relu(self.fc1(x))
    x = self.fc2(x)

    return x.squeeze()

def load_similarity_matrix():
  movie_embeddings = torch.load("models/movie_embeddings.pt")

  similarity_matrix = cosine_similarity(movie_embeddings)

  return similarity_matrix

def recommend_movies_deep(title, n = 5):
  movie_id = movies[movies["title"] == title]["movieId"].values[0]

  idx = movie_map[movie_id]

  similarity_matrix = load_similarity_matrix()

  sim_scores = list(enumerate(similarity_matrix[idx]))
  sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
  sim_scores = sim_scores[1: n + 1]

  movie_indices = [i[0] for i in sim_scores]

  recommended_movie_ids = [reverse_movie_map[i] for i in movie_indices]

  recs = movies[movies["movieId"].isin(recommended_movie_ids)]["title"]

  return recs