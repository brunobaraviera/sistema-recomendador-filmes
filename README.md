# 🎬 Sistema de recomendação de filmes

Este projeto é um sistema de recomendação de filmes desenvolvido com
foco educacional e de portfólio em Machine Learning.

O objetivo é comparar diferentes abordagens de recomendação e demonstrar
uma arquitetura organizada, separando treino e inferência.

---

## 🚀 Tecnologias Utilizadas

- Python
- Pandas
- Scikit-Learn
- PyTorch
- Streamlit

---

## 📌 Modelos Implementados

### 1️⃣ Content-Based Filtering

Recomenda filmes com base na similaridade de gêneros.

Fluxo: - Limpeza dos gêneros - Vetorização com `CountVectorizer` -
Similaridade do cosseno

---

### 2️⃣ Collaborative Filtering (Item-Based)

Recomenda filmes com base no comportamento dos usuários.

Fluxo: - Matriz usuário × filme - Preenchimento de valores ausentes -
Similaridade do cosseno entre filmes

---

### 3️⃣ Deep Learning Recommender (PyTorch)

Modelo baseado em embeddings aprendidos para usuários e filmes.

Arquitetura: - Embedding de usuários - Embedding de filmes - Camadas
fully connected - Função de perda MSE

Treinamento separado do app: - `train_deep_model.py` - Embeddings salvos
em `models/movie_embeddings.pt` - App apenas carrega embeddings para
recomendação

---

## 🏗 Estrutura do Projeto

    sistema-recomendador-filmes/
    │
    ├── app/
    │   └── streamlit_app.py
    │
    ├── models/
    │   └── movie_embeddings.pt
    │
    ├── src/
    │   ├── data_loader.py
    │   ├── content_model.py
    │   ├── collaborative_model.py
    │   ├── deep_model.py
    │   ├── train_deep_model.py
    │
    └── README.md

---

## ▶️ Como Executar

### 1️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 2️⃣ Treinar modelo Deep Learning

```bash
python -m src.train_deep_model
```

Isso criará:

    models/movie_embeddings.pt

### 3️⃣ Rodar a aplicação

```bash
python -m streamlit run app/streamlit_app.py
```

---

## 📊 Comparação de Modelos

O projeto permite comparar três abordagens distintas:

Modelo Base Tipo

---

Content-Based Gêneros Similaridade
Collaborative Ratings de usuários Similaridade
Deep Learning Embeddings aprendidos Rede Neural

---

## 🎯 Objetivo do Projeto

Este projeto demonstra:

- Organização de código em módulos
- Separação entre treino e inferência
- Implementação de múltiplos algoritmos de recomendação
- Pipeline de Machine Learning

---

## 👨‍💻 Autor

Projeto desenvolvido por Bruno Baraviera como parte de portfólio em Machine Learning e Data
Science.
