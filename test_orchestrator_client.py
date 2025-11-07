import asyncio
import websockets
import cv2
import json

# APUNTAMOS AL ORQUESTADOR
WEBSOCKET_URI = "ws://localhost:8000/api/vision/stream/detect"
IMAGE_PATH = "test_face.jpg" # La imagen del coche y la persona

async def run_test():
    try:
        frame = cv2.imread(IMAGE_PATH)
        _, image_bytes = cv2.imencode('.jpg', frame)
        image_bytes = image_bytes.tobytes()

        async with websockets.connect(WEBSOCKET_URI) as websocket:
            print(f"Conectado al ORQUESTADOR en {WEBSOCKET_URI}")
            
            # Enviamos 15 fotogramas para que la estabilización de YOLO funcione
            for i in range(15):
                print(f"-> Enviando fotograma {i+1} al orquestador...")
                await websocket.send(image_bytes)
                
                response_str = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_data = json.loads(response_str)
                
                print(f"<- Fotograma {i+1}: Respuesta COMPLETA recibida del orquestador:")
                print(json.dumps(response_data, indent=2))
                print("-" * 20)
                
                await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
