
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

# Crear carpeta para guardar archivos de salida
os.makedirs("metrics", exist_ok=True)
os.makedirs("models", exist_ok=True)
# --------------------------------------------
# 1. CARGA Y LIMPIEZA DE DATOS
# --------------------------------------------

# Cargar los datasets de ratings y libros
ratings = pd.read_csv("Books_rating.csv")
books = pd.read_csv("books_data.csv")

# Eliminar duplicados en ambos datasets
ratings.drop_duplicates(inplace=True)
books.drop_duplicates(inplace=True)

# Limpiar columnas de texto: quitar espacios y valores nulos
ratings['Title'] = ratings['Title'].astype(str).str.strip()
books['Title'] = books['Title'].astype(str).str.strip()
books['authors'] = books['authors'].astype(str).fillna('')
books['categories'] = books['categories'].astype(str).fillna('')
books['description'] = books['description'].astype(str).fillna('')

# Unir los datasets en una sola tabla por título
merged = pd.merge(ratings, books, on='Title')
# Eliminar registros sin usuario, título o score
merged = merged.dropna(subset=['User_id', 'Title', 'review/score'])

# Filtrar usuarios con al menos 10 calificaciones
user_counts = merged['User_id'].value_counts()
merged = merged[merged['User_id'].isin(user_counts[user_counts >= 10].index)]

# Filtrar libros con al menos 10 calificaciones
book_counts = merged['Title'].value_counts()
merged = merged[merged['Title'].isin(book_counts[book_counts >= 10].index)]

# --------------------------------------------
# 2. NORMALIZACIÓN DE RATINGS POR USUARIO
# --------------------------------------------

# Calcular promedio y desviación estándar del score por usuario
user_stats = merged.groupby("User_id")["review/score"].agg(["mean", "std"]).reset_index()
user_stats.columns = ["User_id", "mean_score", "std_score"]
# Agregar estas estadísticas a cada registro
merged = pd.merge(merged, user_stats, on="User_id")

# Calcular score normalizado: (score - media) / desviación
merged["score_normalized"] = (merged["review/score"] - merged["mean_score"]) / (merged["std_score"] + 1e-6)

# --------------------------------------------
# 3. SPLIT DE DATOS + FILTRADO SEGURO
# --------------------------------------------

# Dividir en entrenamiento y prueba
train_df, test_df = train_test_split(merged, test_size=0.2, random_state=42)

# Asegurar que solo probamos usuarios/libros ya vistos en entrenamiento
train_users = set(train_df['User_id'])
train_titles = set(train_df['Title'])
test_df = test_df[test_df['User_id'].isin(train_users) & test_df['Title'].isin(train_titles)]

# --------------------------------------------
# 4. MODELO COLABORATIVO CON NORMALIZACIÓN
# --------------------------------------------

# Crear matriz usuario × libro con scores normalizados
pivot_train = train_df.pivot_table(index="User_id", columns="Title", values="score_normalized").fillna(0)

# Entrenar SVD (reducción de dimensiones)
svd = TruncatedSVD(n_components=100, random_state=42)
svd_matrix = svd.fit_transform(pivot_train)
reconstructed = np.dot(svd_matrix, svd.components_)

# Desnormalizar predicciones y calcular métricas
y_true, y_pred = [], []
for _, row in test_df.iterrows():
    u, t, r, m, s = row["User_id"], row["Title"], row["review/score"], row["mean_score"], row["std_score"]
    if u in pivot_train.index and t in pivot_train.columns:
        u_idx = pivot_train.index.get_loc(u)
        t_idx = pivot_train.columns.get_loc(t)
        pred_norm = reconstructed[u_idx][t_idx]
        # Desnormalizar predicción y limitar entre 0 y 5
        pred = np.clip(pred_norm * s + m, 0, 5)
        y_true.append(r)
        y_pred.append(pred)

# Calcular métricas de error
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

# Guardar métricas
with open("metrics/hybrid_collaborative_metrics_normalized_filtered.txt", "w") as f:
    f.write(f"RMSE (test set): {rmse:.4f}\n")
    f.write(f"MAE (test set): {mae:.4f}\n")

# Gráfico de dispersión real vs predicho
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_true)), y_true, label="Real", s=2, alpha=0.5)
plt.scatter(range(len(y_pred)), y_pred, label="Predicho", s=2, alpha=0.5, marker='x')
plt.title("Dispersión: Ratings reales vs predi"
          "chos")
plt.xlabel("Índice")
plt.ylabel("Rating")
plt.legend()
plt.tight_layout()
plt.savefig("metrics/hybrid_collaborative_scatter.png")
plt.close()

# Gráfico de error absoluto por predicción
errors = np.abs(np.array(y_true) - np.array(y_pred))
plt.figure(figsize=(10, 5))
plt.plot(errors, linestyle='-', alpha=0.5)
plt.title("Error absoluto por predicción")
plt.xlabel("Índice")
plt.ylabel("Error")
plt.tight_layout()
plt.savefig("metrics/hybrid_collaborative_error_line.png")
plt.close()

# Histograma del error
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=50)
plt.title("Distribución del error absoluto")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("metrics/hybrid_collaborative_error_histogram.png")
plt.close()

# Graficar resultados
plt.figure(figsize=(10, 5))
plt.plot(y_true, label="Real")
plt.plot(y_pred, label="Predicho", linestyle="--")
plt.title("Híbrido Normalizado + Filtrado - Ratings reales vs predichos")
plt.legend()
plt.tight_layout()
plt.savefig("metrics/hybrid_collaborative_predictions_normalized_filtered.png")
plt.close()

# Guardar el modelo colaborativo y estadísticas
joblib.dump((svd, pivot_train, user_stats), "models/modelo_colaborativo_normalizado.pkl")

# --------------------------------------------
# 5. MODELO DE CONTENIDO (TF-IDF + NearestNeighbors)
# --------------------------------------------

# Unir campos de texto relevantes en uno solo
books['combined'] = books['categories'] + " " + books['description'] + " " + books['authors']

# Vectorizar texto (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(books['combined'])

# Crear índice de similitud (Nearest Neighbors con distancia coseno)
nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)

# Índice auxiliar para buscar libros por título
indices = pd.Series(books.index, index=books['Title']).drop_duplicates()

# Generar recomendaciones de ejemplo
sample_output = []
for title in books['Title'].dropna().unique()[:10]:
    idx = indices.get(title)
    if pd.isna(idx): continue
    distances, neighbors = nn.kneighbors(tfidf_matrix[idx])
    similar_titles = books['Title'].iloc[neighbors[0]].tolist()[1:]
    sample_output.append(f"\nLibro: {title}\nSimilares: {similar_titles}")

# Guardar ejemplos de contenido
with open("metrics/hybrid_content_samples.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sample_output))

# Guardar modelo de contenido
joblib.dump((tfidf, tfidf_matrix, nn, books, indices), "models/modelo_contenido_nn.pkl")

# -------------------------------------------------
# MÉTRICA DE SIMILITUD PROMEDIO PARA CONTENIDO
# -------------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity
similarities_all = []
books_titles = books['Title'].dropna().unique()[:100]
normalized_index = {k.strip().lower(): v for k, v in indices.items()}

for input_title in books_titles:
    input_title_norm = input_title.strip().lower()
    if input_title_norm not in normalized_index:
        continue
    idx = normalized_index[input_title_norm]
    vector = tfidf_matrix[idx]
    distances, neighbors = nn.kneighbors(vector)
    recommended_idxs = neighbors[0][1:]
    sim_vectors = tfidf_matrix[recommended_idxs]
    sim_scores = cosine_similarity(vector, sim_vectors)[0]
    if len(sim_scores) > 0:
        similarities_all.append(np.mean(sim_scores))

avg_similarity = np.mean(similarities_all)
with open("metrics/hybrid_content_metrics.txt", "a", encoding="utf-8") as f:
    f.write(f"Similitud promedio entre libros recomendados por contenido: {avg_similarity:.4f}\n")


