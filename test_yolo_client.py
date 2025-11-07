import asyncio
import websockets
import cv2
import json

WEBSOCKET_URI = "ws://localhost:8004/api/v1/yolo/stream"
IMAGE_PATH = "test_yolo_image.jpg" 

async def run_test():
    try:
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print(f"Error: No se pudo cargar la imagen desde {IMAGE_PATH}")
            return
        
        _, image_bytes = cv2.imencode('.jpg', frame)
        image_bytes = image_bytes.tobytes()

        async with websockets.connect(WEBSOCKET_URI) as websocket:
            print(f"Conectado a {WEBSOCKET_URI}")
            
            # Enviamos el mismo fotograma varias veces para probar la estabilización
            print(f"Enviando {HISTORY_LENGTH} fotogramas para llenar el historial del tracker...")
            for i in range(HISTORY_LENGTH):
                await websocket.send(image_bytes)
                try:
                    response_str = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response_str)
                    
                    print(f"<- Fotograma {i+1}: Respuesta recibida.")
                    if response_data.get("objects"):
                        for obj in response_data["objects"]:
                            label = obj.get('label')
                            dist = obj.get('distance_m')
                            color = obj.get('dominant_color_hex')
                            if dist:
                                print(f"  - Objeto estable: {label}, Distancia: {dist:.2f}m, Color: {color}")
                            else:
                                print(f"  - Objeto estable: {label}, Distancia: No calculada, Color: {color}")
                    else:
                        print("  - Aún no hay objetos estables.")

                except asyncio.TimeoutError:
                    print(f"<- Fotograma {i+1}: Sin respuesta (sin objetos estables).")
                
                await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    # Añadimos las constantes aquí para que el script de prueba las conozca
    HISTORY_LENGTH = 15
    asyncio.run(run_test())
