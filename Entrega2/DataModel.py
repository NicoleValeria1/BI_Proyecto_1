from pydantic import BaseModel
from typing import Optional

class DataModel(BaseModel):
    id: Optional[int] = None  # Se genera en el backend
    titulo: str
    descripcion: str
    fecha: str  # Formato "YYYY-MM-DD"

    def get_columns(self):
        return ["id", "titulo", "descripcion", "fecha"]