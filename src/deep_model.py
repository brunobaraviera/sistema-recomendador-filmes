import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import load_movies, load_ratings

movies = load_movies()
ratings = load_ratings()

user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_map = {id: i for i, id in enumerate(user_ids)}
movie_map = {id: i for i, id in enumerate(movie_ids)}
reverse_movie_map = {v: k for k, v in movie_map.items()}

ratings["user_idx"] = ratings["userId"].map(user_map)
ratings["movie_idx"] = ratings["movieId"].map(movie_map)

user_tensor = torch.tensor(ratings["user_idx"].values)
movie_tensor = torch.tensor(ratings["movie_idx"].values)
rating_tensor = torch.tensor(ratings["rating"].values, dtype = torch.float32)

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
  
num_users = len(user_ids)
num_movies = len(movie_ids)

model = RecommenderNet(num_users, num_movies)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
epochs = 5

for epoch in range(epochs):
  optimizer.zero_grad()
  predictions = model(user_tensor, movie_tensor)
  loss = criterion(predictions, rating_tensor)
  loss.backward()
  optimizer.step()
  print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

movie_embeddings = model.movie_embedding.weight.detach()

similarity_matrix = cosine_similarity(movie_embeddings)

def recommend_movies_deep(title, n = 5):
  movie_id = movies[movies["title"] == title]["movieId"].values[0]

  idx = movie_map[movie_id]

  sim_scores = list(enumerate(similarity_matrix[idx]))
  sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
  sim_scores = sim_scores[1: n + 1]

  movie_indices = [i[0] for i in sim_scores]

  recommended_movie_ids = [reverse_movie_map[i] for i in movie_indices]

  recs = movies[movies["movieId"].isin(recommended_movie_ids)]["title"]

  return recs