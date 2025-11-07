import asyncio
from contextlib import asynccontextmanager
from typing import List

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# --- Modelos Pydantic ---
class GestureResponse(BaseModel):
    gestures: List[str]

# --- Modelos de IA y Recursos ---
# Usamos un diccionario para almacenar los modelos cargados.
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al iniciar la aplicación, cargamos el modelo de MediaPipe UNA SOLA VEZ.
    print("Cargando modelo de MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    models["hands_model"] = mp_hands.Hands(
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    )
    print("Modelo cargado exitosamente.")
    yield
    # Al cerrar la aplicación, liberamos los recursos.
    print("Liberando recursos...")
    models["hands_model"].close()
    models.clear()

# --- Instancia de la Aplicación FastAPI ---
app = FastAPI(
    title="AITANA - API Vision Gestures",
    description="Microservicio para detectar gestos en un stream de video vía WebSockets.",
    version="2.0.0",
    lifespan=lifespan,
)

# --- Lógica de Negocio (Función Síncrona) ---
def process_frame_for_gestures(frame_bytes: bytes, hands_model) -> List[str]:
    """
    Función SÍNCRONA y bloqueante que procesa un fotograma para detectar gestos.
    """
    detected_gestures = []
    try:
        # 1. Decodificar los bytes del fotograma a una imagen de OpenCV.
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return []

        # 2. Convertir la imagen a RGB, ya que MediaPipe espera ese formato.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # Optimización: marcar como no escribible.

        # 3. Procesar con MediaPipe.
        results = hands_model.process(rgb_frame)
        
        # 4. Lógica de detección de gestos (ROBUSTA Y CORREGIDA).
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # --- Lógica para "Mano Abierta" ---
                # Obtenemos los landmarks de las puntas y las articulaciones medias de los 4 dedos.
                finger_tips_ids = [
                    mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                    mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                    mp.solutions.hands.HandLandmark.PINKY_TIP,
                ]
                finger_pip_ids = [
                    mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
                    mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
                    mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
                    mp.solutions.hands.HandLandmark.PINKY_PIP,
                ]

                # Verificamos si cada punta de dedo está por encima (menor valor en Y) de su articulación.
                # Esto indica que el dedo está extendido.
                finger_is_extended = [
                    hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y
                    for tip_id, pip_id in zip(finger_tips_ids, finger_pip_ids)
                ]

                # Si todos los dedos están extendidos, consideramos que es una mano abierta.
                if all(finger_is_extended):
                    detected_gestures.append("MANO_ABIERTA")

        return list(set(detected_gestures)) # Devolver gestos únicos

    except Exception as e:
        print(f"Error procesando fotograma: {e}")
        return []

# --- Endpoint WebSocket ---
@app.websocket("/api/v1/gestures/stream")
async def gesture_stream(websocket: WebSocket):
    """
    Endpoint WebSocket para el streaming de detección de gestos.
    - Acepta una conexión.
    - Entra en un bucle para recibir fotogramas (bytes).
    - Procesa cada fotograma en un hilo separado para no bloquear.
    - Envía de vuelta un JSON con los gestos detectados.
    """
    await websocket.accept()
    print("Cliente WebSocket conectado.")
    try:
        while True:
            # 1. Esperar a recibir los bytes de un fotograma del cliente.
            frame_bytes = await websocket.receive_bytes()

            # 2. Ejecutar el procesamiento bloqueante en un hilo separado.
            gestures = await asyncio.to_thread(
                process_frame_for_gestures, frame_bytes, models["hands_model"]
            )

            # 3. Si se detectó algún gesto, enviarlo de vuelta al cliente.
            if gestures:
                response = GestureResponse(gestures=gestures)
                await websocket.send_json(response.dict())

    except WebSocketDisconnect:
        print("Cliente WebSocket desconectado.")
    except Exception as e:
        print(f"Error en la conexión WebSocket: {e}")
    finally:
        # Asegurarse de que la conexión se cierre si hay un error.
        await websocket.close()

@app.get("/health")
def health_check():
    """Endpoint de verificación de estado."""
    return {"status": "ok"}
