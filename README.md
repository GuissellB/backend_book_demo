# 📚 Hybrid Book Recommender System

Este proyecto implementa un sistema de recomendación de libros híbrido que combina filtrado colaborativo (SVD) con recomendación basada en contenido (TF-IDF + Nearest Neighbors). Está desarrollado en Python y expuesto mediante una API REST usando FastAPI.

---

## 📊 Dataset utilizado

Los datos provienen de [Amazon Books Reviews - Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews):
- `Books_rating.csv`: contiene reseñas de usuarios y sus puntuaciones.
- `books_data.csv`: contiene los metadatos de los libros (título, autores, categorías, descripción e imagen).

Se filtraron 5,000 reseñas representativas con variedad de libros para facilitar el entrenamiento y evaluación del modelo.

---

## 🧠 ¿Qué hace el modelo?

### 🔹 `train_recommender_hybrid.py`

Entrena dos modelos distintos y genera métricas:

- **Colaborativo (SVD):**
  - Usa puntuaciones normalizadas por usuario.
  - Aplica reducción de dimensiones con `TruncatedSVD`.
  - Calcula RMSE y MAE para validar precisión.
  - Guarda el modelo en `models/modelo_colaborativo_normalizado.pkl`.

- **Contenido (TF-IDF + kNN):**
  - Vectoriza descripciones, autores y categorías.
  - Aplica `NearestNeighbors` con distancia coseno.
  - Guarda el modelo en `models/modelo_contenido_nn.pkl`.

También se generan visualizaciones y métricas en la carpeta `metrics/`.

---

### 🔹 `endpoints.py`

Expone una API con FastAPI que permite:

- Predecir la calificación estimada para un usuario sobre un libro (`predicted_rating`).
- Devolver libros similares al ingresado basándose en contenido (`content_recommendations`).

#### Endpoint principal:
```
POST /recommend
```

#### Ejemplo de entrada:
```json
{
  "user_id": "A11NCO6YTE4ESA",
  "book_title": "The Fellowship of the Ring (The Lord of the Rings, Part 1)"
}
```

#### Ejemplo de respuesta:
```json
{
  "user_id": "A11NCO6YTE4ESA",
  "book_title": "The Fellowship of the Ring (The Lord of the Rings, Part 1)",
  "predicted_rating": 4.38,
  "content_recommendations": [
    {
      "title": "The Two Towers (The Lord of the Rings, Part 2)",
      "image": "https://images.amazon.com/image1.jpg"
    }
  ]
}
```

---

## 🚀 Instrucciones de ejecución

1. Clonar el repositorio:

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Entrenar modelos:

```bash
python train_recommender_hybrid.py
```

4. Ejecutar API:

```bash
uvicorn endpoints:app --reload
```

5. Visitar Swagger UI:

```
http://localhost:8000/docs
```

---

## 📥 Requisitos

- Python 3.11
- FastAPI
- scikit-learn
- pandas
- numpy
- joblib
- matplotlib

---

## 👥 Colaboradores (estudiantes)

Este proyecto fue desarrollado por:

- Guissell Betancur

---

## 🌿 Ramas utilizadas

Estas son las ramas que se usaron durante el desarrollo:

- `main` — rama principal
- `develop` — para desarrollo continuo
- `staging` — para pruebas previas a producción

---