import asyncio
import numpy as np
from contextlib import asynccontextmanager
from typing import List

import face_recognition
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from PIL import Image
from io import BytesIO

# --- Configuración ---
class Settings(BaseSettings):
    # La URL de conexión a MongoDB. 'mongodb' es el nombre del servicio en docker-compose.
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "aitana_db"

    class Config:
        env_file = ".env"

settings = Settings()

# --- Modelos Pydantic ---
class RegisterFaceResponse(BaseModel):
    message: str
    name: str
    face_id: str = Field(..., alias="_id")

class RecognizeFaceResponse(BaseModel):
    recognized_names: List[str]

# --- Conexión a la Base de Datos ---
# Usamos un diccionario para almacenar la conexión y que sea accesible en la app.
db = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al iniciar la aplicación
    db["client"] = AsyncIOMotorClient(settings.MONGODB_URI)
    db["database"] = db["client"][settings.MONGODB_DB_NAME]
    print("Conexión con MongoDB establecida.")
    yield
    # Al cerrar la aplicación
    db["client"].close()
    print("Conexión con MongoDB cerrada.")

# --- Instancia de la Aplicación FastAPI ---
app = FastAPI(
    title="AITANA - API Vision Faces",
    description="Microservicio para registrar y reconocer rostros.",
    version="2.0.0",
    lifespan=lifespan,
)

# --- Lógica de Negocio (Funciones Síncronas) ---
def process_image_to_encoding(image_bytes: bytes) -> np.ndarray | None:
    """
    Función SÍNCRONA y bloqueante que convierte bytes de imagen en un encoding facial.
    Devuelve el primer encoding encontrado o None si no hay rostros.
    """
    try:
        image = face_recognition.load_image_file(BytesIO(image_bytes))
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            return face_encodings[0] # Devolvemos solo el primer rostro encontrado
        return None
    except Exception as e:
        print(f"Error procesando imagen para encoding: {e}")
        return None

# --- Endpoints de la API ---
@app.post("/api/v1/faces/register")
async def register_face(
    name: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Registra un nuevo rostro en la base de datos.
    Recibe el nombre de la persona y una imagen.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")

    # Ejecutamos la función bloqueante en un hilo separado
    encoding = await asyncio.to_thread(process_image_to_encoding, image_bytes)

    if encoding is None:
        raise HTTPException(status_code=400, detail="No se pudo detectar un rostro en la imagen.")

    # Guardamos en MongoDB usando motor (asíncrono)
    face_document = {
        "name": name,
        "encoding": encoding.tolist(), # MongoDB necesita listas, no arrays de numpy
    }
    result = await db["database"]["faces"].insert_one(face_document)
    
    return {"message": "Rostro registrado exitosamente", "name": name, "_id": str(result.inserted_id)}

@app.post("/api/v1/faces/recognize", response_model=RecognizeFaceResponse)
async def recognize_face(image: UploadFile = File(...)):
    """
    Reconoce rostros en una imagen comparándolos con los registrados en la DB.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")

    unknown_encoding = await asyncio.to_thread(process_image_to_encoding, image_bytes)

    if unknown_encoding is None:
        raise HTTPException(status_code=400, detail="No se pudo detectar un rostro en la imagen para reconocer.")

    known_faces = []
    cursor = db["database"]["faces"].find({})
    async for document in cursor:
        known_faces.append(document)

    if not known_faces:
        return RecognizeFaceResponse(recognized_names=[])

    known_encodings = [np.array(face["encoding"]) for face in known_faces]
    known_names = [face["name"] for face in known_faces]
    
    # La comparación es rápida, pero la mantenemos en un hilo por consistencia
    matches = await asyncio.to_thread(face_recognition.compare_faces, known_encodings, unknown_encoding)
    
    recognized_names = [known_names[i] for i, match in enumerate(matches) if match]

    return RecognizeFaceResponse(recognized_names=list(set(recognized_names))) # Devolvemos nombres únicos

@app.get("/health")
def health_check():
    """Endpoint de verificación de estado."""
    return {"status": "ok"}
