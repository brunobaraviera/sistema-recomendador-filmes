import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
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

  train_df, val_df = train_test_split(
    ratings,
    test_size = 0.2
  )

  train_users = torch.tensor(train_df["user_idx"].values, dtype = torch.long)
  train_movies = torch.tensor(train_df["movie_idx"].values, dtype = torch.long)
  train_ratings = torch.tensor(train_df["rating"].values, dtype = torch.float32)

  val_users = torch.tensor(val_df["user_idx"].values, dtype = torch.long)
  val_movies = torch.tensor(val_df["movie_idx"].values, dtype = torch.long)
  val_ratings = torch.tensor(val_df["rating"].values, dtype = torch.float32)

  num_users = len(user_ids)
  num_movies = len(movie_ids)

  model = RecommenderNet(num_users, num_movies)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr = 0.001)
  epochs = 15

  train_dataset = RatingDataset(train_users, train_movies, train_ratings)
  val_dataset = RatingDataset(val_users, val_movies, val_ratings)

  batch_size = 1024

  train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True
  )

  val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    shuffle = False
  )

  for epoch in range(epochs):
    model.train()
    train_loss = 0

    for users_batch, movies_batch, ratings_batch in train_loader:
      optimizer.zero_grad()
      predictions = model(users_batch, movies_batch)
      loss = criterion(predictions, ratings_batch)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
      for users_batch, movies_batch, ratings_batch in val_loader:
        predicitons = model(users_batch, movies_batch)
        loss = criterion(predicitons, ratings_batch)
        val_loss += loss.item()

    val_loss /= len(val_loader)
    print(
      f"Epoch {epoch + 1}, "
      f"Train Loss: {train_loss:.4f}, "
      f"Val Loss: {val_loss:.4f}"
    )

  movie_embeddings = model.movie_embedding.weight.detach()

  torch.save(movie_embeddings, "models/movie_embeddings.pt")

if __name__ == "__main__":
  main()