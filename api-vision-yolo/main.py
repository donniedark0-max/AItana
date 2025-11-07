import asyncio
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from ultralytics import YOLO
from sklearn.cluster import KMeans

# --- CONSTANTES PARA LA LÓGICA DE NEGOCIO ---
KNOWN_WIDTHS = {"person": 0.5, "laptop": 0.35, "cell phone": 0.07, "bottle": 0.08, "car": 1.8}
FOCAL_LENGTH = 700
HISTORY_LENGTH = 15
CONFIRMATION_THRESHOLD = 5

# --- Modelos Pydantic (Actualizados) ---
class DetectedObject(BaseModel):
    label: str
    confidence: float
    box: List[int]
    distance_m: Optional[float] = None
    dominant_color_hex: Optional[str] = None # NUEVO: Color dominante en formato HEX

class YoloResponse(BaseModel):
    objects: List[DetectedObject]

# --- Modelos de IA y Estado ---
models = {}
detection_tracker = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Determinando dispositivo de cómputo...")
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Aceleración MPS (Apple Silicon GPU) detectada y disponible.")
    else:
        print("⚠️ MPS no disponible, usando CPU.")
    
    models["device"] = device
    
    print("Cargando modelo YOLOv11...")
    model_path = "/app/models/yolov11x.pt"
    models["yolo_model"] = YOLO(model_path)
    models["yolo_model"].to(device)
    print(f"Modelo YOLO cargado exitosamente en el dispositivo: '{device}'.")
    
    yield
    
    print("Liberando recursos...")
    models.clear()
    detection_tracker.clear()

# --- Instancia de la Aplicación FastAPI ---
app = FastAPI(
    title="AITANA - API Vision YOLO (Enriquecido)",
    description="Microservicio para detectar objetos con estabilización, distancia y color.",
    version="2.2.0",
    lifespan=lifespan,
)

# --- Lógica de Negocio (Funciones Síncronas) ---

def calculate_distance(known_width_m: float, focal_length_px: float, perceived_width_px: int) -> float:
    if perceived_width_px == 0: return float('inf')
    return (known_width_m * focal_length_px) / perceived_width_px

def get_dominant_color(image_crop: np.ndarray) -> str:
    """
    NUEVA FUNCIÓN: Encuentra el color dominante en un recorte de imagen usando K-Means.
    """
    try:
        # Remodelar la imagen para que sea una lista de píxeles
        pixels = image_crop.reshape((-1, 3))
        pixels = np.float32(pixels)

        # Usar K-Means para encontrar el cluster de color principal (el color dominante)
        kmeans = KMeans(n_clusters=1, n_init=1)
        kmeans.fit(pixels)
        
        # El centro del cluster es el color dominante en formato BGR
        dominant_color_bgr = kmeans.cluster_centers_[0].astype(int)
        
        # Convertir de BGR a formato HEX para una fácil transmisión
        return f"#{dominant_color_bgr[2]:02x}{dominant_color_bgr[1]:02x}{dominant_color_bgr[0]:02x}"
    except Exception:
        return None

def process_frame_for_yolo(frame_bytes: bytes, yolo_model, device: str) -> List[DetectedObject]:
    try:
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None: return []

        results = yolo_model(frame, device=device, verbose=False)
        result = results[0]
        
        current_frame_labels = {result.names[int(box.cls)] for box in result.boxes}
        all_known_labels = set(detection_tracker.keys()) | current_frame_labels
        
        for label in all_known_labels:
            detection_tracker[label].append(label in current_frame_labels)

        stable_objects = []
        for label, history in detection_tracker.items():
            if sum(history) >= CONFIRMATION_THRESHOLD:
                for box in result.boxes:
                    if result.names[int(box.cls)] == label:
                        confidence = float(box.conf)
                        x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                        
                        distance = None
                        if label in KNOWN_WIDTHS:
                            distance = calculate_distance(KNOWN_WIDTHS[label], FOCAL_LENGTH, x2 - x1)
                        
                        # NUEVO: Extraer color dominante del recorte del objeto
                        object_crop = frame[y1:y2, x1:x2]
                        color_hex = get_dominant_color(object_crop)
                        
                        stable_objects.append(
                            DetectedObject(
                                label=label, 
                                confidence=confidence, 
                                box=[x1, y1, x2, y2], 
                                distance_m=distance,
                                dominant_color_hex=color_hex
                            )
                        )
                        break
        return stable_objects
    except Exception as e:
        print(f"Error procesando fotograma con YOLO: {e}")
        return []

# --- Endpoint WebSocket (sin cambios) ---
@app.websocket("/api/v1/yolo/stream")
async def yolo_stream(websocket: WebSocket):
    await websocket.accept()
    print("Cliente YOLO WebSocket conectado.")
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            objects = await asyncio.to_thread(
                process_frame_for_yolo, frame_bytes, models["yolo_model"], models["device"]
            )
            if objects:
                response = YoloResponse(objects=objects)
                await websocket.send_json(response.dict())
    except WebSocketDisconnect:
        print("Cliente YOLO WebSocket desconectado.")
    except Exception as e:
        print(f"Error en la conexión YOLO WebSocket: {e}")
    finally:
        await websocket.close()

@app.get("/health")
def health_check():
    """Endpoint de verificación de estado."""
    return {"status": "ok"}
