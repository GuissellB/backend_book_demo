import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Crear carpetas para salidas
os.makedirs("metrics", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --------------------------------------------
# 1. CARGA Y LIMPIEZA DE DATOS
# --------------------------------------------
ratings = pd.read_csv("Books_rating.csv")
books = pd.read_csv("books_data.csv")

ratings.drop_duplicates(inplace=True)
books.drop_duplicates(inplace=True)

ratings['Title'] = ratings['Title'].astype(str).str.strip()
books['Title'] = books['Title'].astype(str).str.strip()
books['authors'] = books['authors'].astype(str).fillna('')
books['categories'] = books['categories'].astype(str).fillna('')
books['description'] = books['description'].astype(str).fillna('')

merged = pd.merge(ratings, books, on='Title')
merged = merged.dropna(subset=['User_id', 'Title', 'review/score'])

# --------------------------------------------
# 2. NORMALIZACIÓN DE RATINGS POR USUARIO
# --------------------------------------------
user_stats = merged.groupby("User_id")["review/score"].agg(["mean", "std"]).reset_index()
user_stats.columns = ["User_id", "mean_score", "std_score"]
merged = pd.merge(merged, user_stats, on="User_id")

merged["score_normalized"] = (merged["review/score"] - merged["mean_score"]) / (merged["std_score"] + 1e-6)

# --------------------------------------------
# 3. SPLIT DE DATOS + FILTRADO SEGURO
# --------------------------------------------
train_df, test_df = train_test_split(merged, test_size=0.2, random_state=42)
train_users = set(train_df['User_id'])
train_titles = set(train_df['Title'])
test_df = test_df[test_df['User_id'].isin(train_users) & test_df['Title'].isin(train_titles)]

# --------------------------------------------
# 4. MODELO COLABORATIVO CON NORMALIZACIÓN
# --------------------------------------------
pivot_train = train_df.pivot_table(index="User_id", columns="Title", values="score_normalized").fillna(0)

svd = TruncatedSVD(n_components=50, random_state=42)
svd_matrix = svd.fit_transform(pivot_train)
reconstructed = np.dot(svd_matrix, svd.components_)

y_true, y_pred = [], []
for _, row in test_df.iterrows():
    u, t, r, m, s = row["User_id"], row["Title"], row["review/score"], row["mean_score"], row["std_score"]
    if u in pivot_train.index and t in pivot_train.columns:
        u_idx = pivot_train.index.get_loc(u)
        t_idx = pivot_train.columns.get_loc(t)
        pred_norm = reconstructed[u_idx][t_idx]
        pred = np.clip(pred_norm * s + m, 0, 5)
        y_true.append(r)
        y_pred.append(pred)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

with open("metrics/hybrid_collaborative_metrics_normalized_filtered.txt", "w") as f:
    f.write(f"RMSE (test set): {rmse:.4f}\n")
    f.write(f"MAE (test set): {mae:.4f}\n")

plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_true)), y_true, label="Real", s=2, alpha=0.5)
plt.scatter(range(len(y_pred)), y_pred, label="Predicho", s=2, alpha=0.5, marker='x')
plt.title("Ratings reales vs predichos")
plt.xlabel("Índice")
plt.ylabel("Rating")
plt.legend()
plt.tight_layout()
plt.savefig("metrics/hybrid_collaborative_scatter.png")
plt.close()

errors = np.abs(np.array(y_true) - np.array(y_pred))
plt.figure(figsize=(10, 5))
plt.plot(errors, linestyle='-', alpha=0.5)
plt.title("Error absoluto por predicción")
plt.xlabel("Índice")
plt.ylabel("Error")
plt.tight_layout()
plt.savefig("metrics/hybrid_collaborative_error_line.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=50)
plt.title("Distribución del error absoluto")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("metrics/hybrid_collaborative_error_histogram.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(y_true, label="Real")
plt.plot(y_pred, label="Predicho", linestyle="--")
plt.title("Ratings reales vs predichos")
plt.legend()
plt.tight_layout()
plt.savefig("metrics/hybrid_collaborative_predictions_normalized_filtered.png")
plt.close()

joblib.dump((svd, pivot_train, user_stats), "models/modelo_colaborativo_normalizado.pkl")

# --------------------------------------------
# 5. MODELO DE CONTENIDO
# --------------------------------------------
books['combined'] = books['categories'] + " " + books['description'] + " " + books['authors']
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(books['combined'])

nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)
indices = pd.Series(books.index, index=books['Title']).drop_duplicates()

sample_output = []
for title in books['Title'].dropna().unique()[:10]:
    idx = indices.get(title)
    if pd.isna(idx): continue
    distances, neighbors = nn.kneighbors(tfidf_matrix[idx])
    similar_titles = books['Title'].iloc[neighbors[0]].tolist()[1:]
    sample_output.append(f"\nLibro: {title}\nSimilares: {similar_titles}")

with open("metrics/hybrid_content_samples.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sample_output))

joblib.dump((tfidf, tfidf_matrix, nn, books, indices), "models/modelo_contenido_nn.pkl")

similarities_all = []
books_titles = books['Title'].dropna().unique()[:100]
normalized_index = {k.strip().lower(): v for k, v in indices.items()}

for input_title in books_titles:
    input_title_norm = input_title.strip().lower()
    if input_title_norm not in normalized_index:
        continue
    idx = normalized_index[input_title_norm]
    vector = tfidf_matrix[idx]
    _, neighbors = nn.kneighbors(vector)
    recommended_idxs = neighbors[0][1:]
    sim_vectors = tfidf_matrix[recommended_idxs]
    sim_scores = cosine_similarity(vector, sim_vectors)[0]
    if len(sim_scores) > 0:
        similarities_all.append(np.mean(sim_scores))

avg_similarity = np.mean(similarities_all)
with open("metrics/hybrid_content_metrics.txt", "a", encoding="utf-8") as f:
    f.write(f"Similitud promedio entre libros recomendados por contenido: {avg_similarity:.4f}\n")
