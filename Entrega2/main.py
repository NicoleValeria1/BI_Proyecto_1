from typing import Optional

from fastapi import FastAPI

import pandas as pd 
from joblib import load
from DataModel import DataModel
modelo = load("modelo_fake_news.joblib")

app = FastAPI()
# Cargar el modelo entrenado


@app.get("/")
def read_root():
   return {"message": "API de detección de noticias falsas"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.post("/predecir/")
def predecir_noticia(noticia: DataModel):
    texto = noticia.titulo + " " + noticia.descripcion
    probabilidad = modelo.predict_proba([texto])[0, 1]
    prediccion = int(probabilidad >= 0.6)  # Umbral definido
    return {
        "titulo": noticia.titulo, 
        "descripcion": noticia.descripcion, 
        "prediccion": prediccion, 
        "probabilidad": probabilidad
    }