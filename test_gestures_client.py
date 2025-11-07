import asyncio
import websockets
import cv2

WEBSOCKET_URI = "ws://localhost:8003/api/v1/gestures/stream"
IMAGE_PATH = "test_image.jpg" # Asegúrate de tener una imagen con una mano visible aquí

async def run_test():
    try:
        # Cargar y codificar la imagen
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print(f"Error: No se pudo cargar la imagen desde {IMAGE_PATH}")
            return
        
        _, image_bytes = cv2.imencode('.jpg', frame)
        image_bytes = image_bytes.tobytes()

        async with websockets.connect(WEBSOCKET_URI) as websocket:
            print(f"Conectado a {WEBSOCKET_URI}")
            
            # Enviar la imagen 10 veces para simular un stream
            for i in range(10):
                print(f"-> Enviando fotograma {i+1}...")
                await websocket.send(image_bytes)
                
                # Esperar y recibir la respuesta
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    print(f"<- Respuesta recibida: {response}")
                except asyncio.TimeoutError:
                    print("<- No se recibió respuesta (sin gestos detectados).")

                await asyncio.sleep(0.5) # Simular un delay entre fotogramas

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Conexión cerrada: {e}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
