import streamlit as st
import pandas as pd
from src.data_loader import load_movies

def disable():
  st.session_state.loaded = True

def reset():
  st.session_state.loaded = False
  st.session_state.selection = None

st.set_page_config(
  page_title = "Recomendação de filmes",
  layout = "centered"
)

st.title("Recomendação de Filmes", text_alignment = "center")

if not "selection" in st.session_state:
  st.session_state.selection = None

if not "loaded" in st.session_state:
  st.session_state.loaded = False

with st.spinner("Carregando..."):
  movies = load_movies()

movie_titles = movies["title"].sort_values().unique()

selected_movie = st.selectbox(
  "Escolha um filme",
  movie_titles,
  index = None,
  disabled = st.session_state.loaded,
  key = "selection",
  placeholder = "Selecione uma opção"
)

request_button = st.button("Ver recomendações", disabled = st.session_state.loaded, on_click = disable)

if request_button:
  with st.spinner("Carregando..."):
    from src.content_model import recommend_movies
    from src.collaborative_model import recommend_movies_collaborative
    from src.deep_model import recommend_movies_deep
    
  st.header(f"Recomendações para: {selected_movie}")

  st.markdown("### Modelo de recomendação por gênero")
  with st.spinner("Carregando..."):
    recs = recommend_movies(selected_movie)
  for movie in recs:
    st.write(movie)

  st.markdown("### Modelo de recomendação colaborativo")
  with st.spinner("Carregando..."):
    recs = recommend_movies_collaborative(selected_movie)
  for movie in recs:
    st.write(movie)

  st.markdown("### Modelo de recomendação com Deep Learning")
  with st.spinner("Carregando..."):
    recs = recommend_movies_deep(selected_movie)
  for movie in recs:
    st.write(movie)

  reset_button = st.button("Escolher outro filme", on_click = reset)