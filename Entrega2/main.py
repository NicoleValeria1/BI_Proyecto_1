from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd 
from joblib import load
from DataModel import DataModel
modelo = load("modelo_fake_news.joblib")

app = FastAPI()
# Cargar el modelo entrenado

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes cambiar "*" por dominios específicos
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (OPTIONS incluido)
    allow_headers=["*"],  # Permitir todos los headers
)

@app.get("/")
def read_root():
   return {"message": "API de detección de noticias falsas"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

contador_id = 1  # Variable global para generar IDs únicos

@app.post("/predecir/")
def predecir_noticia(noticias: list[DataModel]):
    global contador_id
    resultados = []
    
    for noticia in noticias:
        if noticia.id is None:
            noticia.id = contador_id
            contador_id += 1  # Incrementar ID
        
        texto = noticia.titulo + " " + noticia.descripcion
        probabilidad = modelo.predict_proba([texto])[0, 1]
        prediccion = int(probabilidad >= 0.6)

        resultados.append({
            "id": noticia.id,
            "titulo": noticia.titulo,
            "descripcion": noticia.descripcion,
            "fecha": noticia.fecha,
            "prediccion": prediccion,
            "probabilidad": probabilidad
        })
    
    return resultados