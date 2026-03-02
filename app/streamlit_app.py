import streamlit as st
import pandas as pd
from src.data_loader import load_movies
from src.content_model import recommend_movies
from src.collaborative_model import recommend_movies_collaborative
from src.deep_model import recommend_movies_deep

st.set_page_config(
  page_title = "Recomendador de filmes",
  layout = "centered"
)

st.title("Sistema de Recomendação de Filmes")

movies = load_movies()

movie_titles = movies["title"].sort_values().unique()

selected_movie = st.selectbox(
  "Escolha um filme",
  movie_titles
)

if st.button("Recomendar"):
  st.subheader(f"Recomendações para: {selected_movie}")

  st.markdown("### Modelo de recomendação por gênero")
  recs = recommend_movies(selected_movie)
  for movie in recs:
    st.write(movie)

  st.markdown("### Modelo de recomendação colaborativo")
  recs = recommend_movies_collaborative(selected_movie)
  for movie in recs:
    st.write(movie)

  st.markdown("### Modelo de recomendação com Deep Learning")
  recs = recommend_movies_deep(selected_movie)
  for movie in recs:
    st.write(movie)