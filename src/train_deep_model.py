import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.data_loader import load_movies, load_ratings
from src.deep_model import RecommenderNet

class RatingDataset(Dataset):
  def __init__(self, users, movies, ratings):
    self.users = users
    self.movies = movies
    self.ratings = ratings

  def __len__(self):
    return len(self.ratings)
  
  def __getitem__(self, idx):
    return (
      self.users[idx],
      self.movies[idx],
      self.ratings[idx]
    )

def main():

  movies = load_movies()
  ratings = load_ratings()

  user_ids = ratings["userId"].unique()
  movie_ids = ratings["movieId"].unique()

  user_map = {id: i for i, id in enumerate(user_ids)}
  movie_map = {id: i for i, id in enumerate(movie_ids)}

  ratings["user_idx"] = ratings["userId"].map(user_map)
  ratings["movie_idx"] = ratings["movieId"].map(movie_map)

  user_tensor = torch.tensor(ratings["user_idx"].values, dtype = torch.long)
  movie_tensor = torch.tensor(ratings["movie_idx"].values, dtype = torch.long)
  rating_tensor = torch.tensor(ratings["rating"].values, dtype = torch.float32)

  num_users = len(user_ids)
  num_movies = len(movie_ids)

  model = RecommenderNet(num_users, num_movies)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr = 0.001)
  epochs = 30
  dataset = RatingDataset(user_tensor, movie_tensor, rating_tensor)
  batch_size = 1024
  dataloader = DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = True
  )

  for epoch in range(epochs):
    total_loss = 0

    for users_batch, movies_batch, ratings_batch in dataloader:
      optimizer.zero_grad()
      predictions = model(users_batch, movies_batch)
      loss = criterion(predictions, ratings_batch)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

  movie_embeddings = model.movie_embedding.weight.detach()

  torch.save(movie_embeddings, "models/movie_embeddings.pt")

if __name__ == "__main__":
  main()