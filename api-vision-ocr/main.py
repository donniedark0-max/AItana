import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import pytesseract
from io import BytesIO

# --- Modelos Pydantic ---
# Definen la estructura de los datos de entrada y salida.
# Esto nos da validación automática y una excelente documentación en /docs.

class OCRResponse(BaseModel):
    """Modelo de respuesta para la extracción de texto."""
    text: str

# --- Instancia de la Aplicación FastAPI ---
app = FastAPI(
    title="AITANA - API Vision OCR",
    description="Microservicio para realizar Reconocimiento Óptico de Caracteres (OCR) en imágenes.",
    version="2.0.0",
)

# --- Lógica de Negocio (Función Síncrona) ---

def run_tesseract(image_bytes: bytes) -> str:
    """
    Ejecuta Pytesseract en un bloque de bytes de imagen.
    Esta es una función SÍNCRONA y bloqueante (intensiva en CPU).
    Debe ser ejecutada en un hilo separado para no bloquear el event loop de FastAPI.
    """
    try:
        # Usamos Pillow para abrir la imagen desde los bytes en memoria
        image = Image.open(BytesIO(image_bytes))
        
        # Ejecutamos tesseract. Especificamos 'spa' para español.
        # El motor de Tesseract se instaló en el Dockerfile.
        detected_text = pytesseract.image_to_string(image, lang='spa')
        
        return detected_text.strip()
    except Exception as e:
        # Capturamos cualquier error durante el procesamiento de la imagen
        # y lo relanzamos como una excepción que nuestro endpoint manejará.
        print(f"Error en Tesseract: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen con Tesseract: {e}")


# --- Endpoints de la API ---

@app.post("/api/v1/ocr", response_model=OCRResponse)
async def perform_ocr(image: UploadFile = File(...)):
    """
    Recibe una imagen, extrae el texto usando Tesseract OCR y lo devuelve.
    
    La operación de OCR es bloqueante, por lo que se ejecuta en un hilo
    externo usando `asyncio.to_thread` para mantener el servidor reactivo.
    """
    # 1. Leer los bytes de la imagen subida. Esta parte es asíncrona.
    image_bytes = await image.read()

    # 2. Validar que el archivo no esté vacío.
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No se proporcionó ninguna imagen.")

    # 3. Ejecutar la función bloqueante en un hilo separado.
    # `asyncio.to_thread` delega la función `run_tesseract` a un
    # thread pool, y esperamos (await) a que termine sin bloquear
    # el hilo principal del event loop.
    try:
        extracted_text = await asyncio.to_thread(run_tesseract, image_bytes)
        
        # 4. Devolver el resultado en el formato definido por OCRResponse.
        return OCRResponse(text=extracted_text)
    
    except HTTPException as e:
        # Si run_tesseract lanzó una HTTPException, la reenviamos.
        raise e
    except Exception as e:
        # Captura cualquier otro error inesperado.
        raise HTTPException(status_code=500, detail=f"Ocurrió un error inesperado: {e}")

@app.get("/health")
def health_check():
    """Endpoint de verificación de estado."""
    return {"status": "ok"}
