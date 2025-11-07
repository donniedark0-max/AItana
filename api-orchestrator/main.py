import asyncio
import json
from contextlib import asynccontextmanager
from typing import Dict, Optional

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# --- Configuración (sin cambios) ---
class Settings(BaseSettings):
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "aitana_db"
    GEMINI_API_KEY: str = "AIzaSyCUwqoYe4td_he93p3p8yhRSPdwQCYGPfg"
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
settings = Settings()

# --- Clientes, Conexiones y Estado Global ---
db = {}
http_client: Dict[str, httpx.AsyncClient] = {}

# NUEVO: Mecanismo de captura con Evento y Cola
frame_capture_trigger: Optional[asyncio.Event] = None
frame_queue = asyncio.Queue(maxsize=1)

# URLs internas (sin cambios)
YOLO_WORKER_URL = "ws://api-vision-yolo:8000/api/v1/yolo/stream"
GESTURES_WORKER_URL = "ws://api-vision-gestures:8000/api/v1/gestures/stream"
FACES_WORKER_URL = "http://api-vision-faces:8000/api/v1/faces/register"

@asynccontextmanager
async def lifespan(app: FastAPI):
    db["client"] = AsyncIOMotorClient(settings.MONGODB_URI)
    db["database"] = db["client"][settings.MONGODB_DB_NAME]
    http_client["client"] = httpx.AsyncClient(timeout=10.0)
    print("✅ Orquestador iniciado.")
    yield
    db["client"].close()
    await http_client["client"].aclose()
    print("❌ Orquestador detenido.")

# --- Modelos Pydantic (sin cambios) ---
class RegisterFaceRequest(BaseModel): name: str
class RegisterFaceResponse(BaseModel): message: str; name: str; face_id: str

app = FastAPI(
    title="AITANA 2.0 - API Orchestrator",
    description="El cerebro central de AITANA.",
    version="2.2.0",
    lifespan=lifespan,
)

# --- Lógica de Orquestación (CORREGIDA) ---
async def process_on_worker(ws: websockets.WebSocketClientProtocol, frame_bytes: bytes) -> str:
    try:
        await ws.send(frame_bytes)
        return await asyncio.wait_for(ws.recv(), timeout=5.0)
    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed): return "{}"
    except Exception: return "{}"

@app.websocket("/api/vision/stream/detect")
async def vision_stream_detect(client_ws: WebSocket):
    global frame_capture_trigger
    await client_ws.accept()
    print("Cliente de visión conectado.")
    yolo_ws, gestures_ws = None, None
    try:
        yolo_ws = await websockets.connect(YOLO_WORKER_URL)
        gestures_ws = await websockets.connect(GESTURES_WORKER_URL)
        print("Orquestador conectado a workers.")
        while True:
            frame_bytes = await client_ws.receive_bytes()

            # Si hay una solicitud de captura pendiente, la atendemos
            if frame_capture_trigger and not frame_capture_trigger.is_set():
                if not frame_queue.full():
                    await frame_queue.put(frame_bytes)
                    frame_capture_trigger.set() # "Levantamos la bandera"

            # El streaming a los workers continúa normalmente
            yolo_task = asyncio.create_task(process_on_worker(yolo_ws, frame_bytes))
            gestures_task = asyncio.create_task(process_on_worker(gestures_ws, frame_bytes))
            results = await asyncio.gather(yolo_task, gestures_task)
            
            yolo_data = json.loads(results[0] or "{}")
            gestures_data = json.loads(results[1] or "{}")
            await client_ws.send_json({"vision": yolo_data, "gestures": gestures_data})
    except WebSocketDisconnect: print("Cliente de visión desconectado.")
    except Exception as e: print(f"Error en stream de visión: {e}")
    finally:
        if yolo_ws: await yolo_ws.close()
        if gestures_ws: await gestures_ws.close()

@app.post("/api/v1/faces/register/start", response_model=RegisterFaceResponse)
async def start_face_registration(request: RegisterFaceRequest = Body(...)):
    global frame_capture_trigger
    print(f"Iniciando registro para: {request.name}")
    
    # Creamos un nuevo evento para esta solicitud de captura
    frame_capture_trigger = asyncio.Event()
    
    try:
        print("Esperando señal de captura de fotograma...")
        await asyncio.wait_for(frame_capture_trigger.wait(), timeout=5.0)
        
        print("Señal recibida. Tomando fotograma de la cola...")
        frame_bytes = frame_queue.get_nowait()
        
        files = {'image': ('capture.jpg', frame_bytes, 'image/jpeg')}
        data = {'name': request.name}

        client = http_client["client"]
        print(f"Enviando fotograma a {FACES_WORKER_URL}...")
        response = await client.post(FACES_WORKER_URL, files=files, data=data)

        if response.status_code == 200:
            worker_response = response.json()
            print(f"Registro exitoso en worker: {worker_response}")
            return RegisterFaceResponse(
                message="Rostro registrado exitosamente",
                name=worker_response.get("name"),
                face_id=worker_response.get("_id")
            )
        else:
            print(f"Error del worker: {response.status_code} {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.json())

    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="No se pudo capturar un fotograma. ¿Stream activo?")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")
    finally:
        # Limpiamos el trigger para la siguiente solicitud
        frame_capture_trigger = None

@app.get("/health")
def health_check(): return {"status": "ok"}
