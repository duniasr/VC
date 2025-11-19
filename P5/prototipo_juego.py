import cv2
import numpy as np
import random
from deepface import DeepFace

print("Iniciando Prototipo 3: Face Jump (Versión DeepFace) 🎮")
print("- Mueve la cabeza ARRIBA y ABAJO.")
print("- Controlamos usando la PUNTA DE LA NARIZ.")
print("- Pulsa ESC para salir.")

# --- 1. Configuración ---
# Usamos 'opencv' como backend dentro de DeepFace porque es el más rápido para juegos.
# Podrías probar 'ssd' o 'mtcnn', pero el juego irá más lento.
BACKEND = "opencv" 

# --- 2. Variables del Juego ---
width, height = 640, 480
player_x = 100          
player_radius = 20      
obstacle_width = 50
obstacle_speed = 10
obstacle_gap = 200      
score = 0
game_over = False
obstacles = []

def reset_game():
    global obstacles, score, game_over, obstacle_speed
    obstacles = []
    score = 0
    obstacle_speed = 10
    game_over = False
    obstacles.append([width, random.randint(50, height - 50 - obstacle_gap)])

reset_game()

# --- 3. Bucle Principal ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    
    player_y = height // 2 # Posición por defecto

    try:
        # --- DETECCIÓN CON DEEPFACE ---
        # extract_faces nos da la caja y también los "landmarks" (ojos, nariz, boca)
        faces = DeepFace.extract_faces(
            img_path=frame, 
            detector_backend=BACKEND, 
            enforce_detection=False
        )

        # Si encontramos alguna cara válida...
        for face in faces:
            # Filtramos detecciones malas (confianza baja)
            if face.get('confidence', 1) < 0.5: 
                continue
            
            # Intentamos obtener la nariz
            landmarks = face.get('landmarks')
            
            if landmarks is not None and 'nose' in landmarks and landmarks['nose'] is not None:
                # ¡Tenemos nariz! Usamos su posición Y
                nose_point = landmarks['nose']
                player_y = int(nose_point[1])
                
                # Dibujamos un puntito rojo en la nariz real para que veas qué rastrea
                cv2.circle(frame, (int(nose_point[0]), int(nose_point[1])), 5, (0, 0, 255), -1)
                break # Solo usamos la primera cara que tenga nariz
            
            else:
                # Si el backend no devuelve nariz, usamos el centro de la cara (Plan B)
                area = face['facial_area']
                player_y = int(area['y'] + area['h'] / 2)
                break

    except Exception as e:
        pass # Si DeepFace falla en un frame, no pasa nada

    # --- LÓGICA DEL JUEGO (Igual que antes) ---
    if not game_over:
        for obs in obstacles:
            obs[0] -= obstacle_speed
        
        if obstacles[-1][0] < width - 300:
            obstacles.append([width, random.randint(50, height - 50 - obstacle_gap)])

        if obstacles[0][0] < -obstacle_width:
            obstacles.pop(0)
            score += 1
            if score % 5 == 0: obstacle_speed += 2

        # Detección de Colisiones
        hitbox_margin = 10
        for obs in obstacles:
            obs_x, gap_y = obs
            if (player_x + player_radius - hitbox_margin > obs_x and 
                player_x - player_radius + hitbox_margin < obs_x + obstacle_width):
                if (player_y - player_radius + hitbox_margin < gap_y) or \
                   (player_y + player_radius - hitbox_margin > gap_y + obstacle_gap):
                    game_over = True

    # --- DIBUJADO ---
    for obs in obstacles:
        obs_x, gap_y = obs
        cv2.rectangle(frame, (obs_x, 0), (obs_x + obstacle_width, gap_y), (0, 0, 255), -1)
        cv2.rectangle(frame, (obs_x, gap_y + obstacle_gap), (obs_x + obstacle_width, height), (0, 0, 255), -1)

    color_player = (255, 0, 0) if not game_over else (0, 0, 150)
    cv2.circle(frame, (player_x, player_y), player_radius, color_player, -1)
    cv2.circle(frame, (player_x, player_y), player_radius, (255, 255, 255), 2)

    cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    if game_over:
        cv2.putText(frame, "GAME OVER", (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.putText(frame, "ESPACIO para reiniciar", (width//2 - 180, height//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Prototipo 3 - Face Jump (DeepFace)', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    if key == 32: reset_game()

cap.release()
cv2.destroyAllWindows()