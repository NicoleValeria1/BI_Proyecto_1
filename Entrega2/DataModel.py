from pydantic import BaseModel

class DataModel(BaseModel):

    # Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    ID:int
    Titulo: str
    Descripcion: str
    Fecha:int
   
   

    #Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return ["ID","Titulo", "Descripcion", "Fecha"]

