
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Cargar modelos y datos
svd, pivot_train, user_stats = joblib.load("models/modelo_colaborativo_normalizado.pkl")
tfidf, tfidf_matrix, nn, books_df, indices = joblib.load("models/modelo_contenido_nn.pkl")
books_data = pd.read_csv("books_data.csv")

# Modelo de entrada
class RecommendationRequest(BaseModel):
    user_id: str
    book_title: str

# Endpoint
@app.post("/recommend")
def recommend(data: RecommendationRequest):
    user_id = data.user_id
    book_title = data.book_title
    book_title_normalized = book_title.strip().lower()

    predicted_rating = None
    recommendations = []

    # Predicción colaborativa
    if user_id in pivot_train.index and book_title in pivot_train.columns:
        try:
            u_idx = pivot_train.index.get_loc(user_id)
            t_idx = pivot_train.columns.get_loc(book_title)
            user_vector = pivot_train.iloc[u_idx].values.reshape(1, -1)
            pred_norm = np.dot(svd.transform(user_vector), svd.components_)[0][t_idx]
            user_mean = user_stats[user_stats["User_id"] == user_id]["mean_score"].values[0]
            user_std = user_stats[user_stats["User_id"] == user_id]["std_score"].values[0]
            predicted_rating = float(np.clip(pred_norm * user_std + user_mean, 0, 5))
        except:
            predicted_rating = None

    # Recomendaciones por contenido
    normalized_indices = {k.strip().lower(): v for k, v in indices.items()}
    if book_title_normalized not in normalized_indices:
        raise HTTPException(status_code=404, detail="Libro no encontrado para recomendación de contenido")

    idx = normalized_indices[book_title_normalized]
    distances, neighbors = nn.kneighbors(tfidf_matrix[idx])
    similar_titles = books_df["Title"].iloc[neighbors[0]].tolist()
    similar_titles = [b for b in similar_titles if b.strip().lower() != book_title_normalized]

    for title in similar_titles:
        image = books_data.loc[books_data["Title"].str.strip().str.lower() == title.strip().lower(), "image"]
        image_url = None
        if not image.empty:
            val = image.values[0]
            if pd.notna(val):
                image_url = val
        recommendations.append({"title": title, "image": image_url})

    return {
        "user_id": user_id,
        "book_title": book_title,
        "predicted_rating": predicted_rating,
        "content_recommendations": recommendations
    }
