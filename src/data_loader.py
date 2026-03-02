import pandas as pd
from pathlib import Path

DATA_PATH = Path("data")

def load_movies():
  movies = pd.read_csv(DATA_PATH / "movies.csv")
  return movies

def load_ratings():
  ratings = pd.read_csv(DATA_PATH / "ratings.csv")
  return ratings

def load_tags():
  tags = pd.read_csv(DATA_PATH / "tags.csv")
  return tags