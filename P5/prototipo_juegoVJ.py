import cv2
import numpy as np
import random

print("Iniciando Prototipo 3: Face Jump (Versión OpenCV) 🎮")
print("- Mueve la cabeza ARRIBA y ABAJO para esquivar.")
print("- Pulsa ESPACIO para reiniciar si pierdes.")
print("- Pulsa ESC para salir.")

# --- 1. Configuración de OpenCV (Haar Cascade) ---
# Cargamos el detector clásico pre-entrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. Variables del Juego ---
width, height = 640, 480
player_x = 100          
player_radius = 20      
obstacle_width = 50
obstacle_speed = 10
obstacle_gap = 200      # Espacio vertical libre
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

    # Espejo y redimensionar
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    
    # OpenCV necesita escala de grises para detectar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de caras
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    player_y = height // 2 # Posición por defecto

    # Si detecta alguna cara, usamos la primera para controlar al jugador
    if len(faces) > 0:
        (x, y, w, h) = faces[0] # Tomamos la primera cara
        
        # El centro de la cara será nuestro "joystick"
        player_y = int(y + h / 2)
        
        # Dibujar un cuadrado sutil alrededor de la cara detectada (opcional)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # --- LÓGICA DEL JUEGO ---
    if not game_over:
        # 1. Mover obstáculos
        for obs in obstacles:
            obs[0] -= obstacle_speed
        
        # 2. Generar nuevos obstáculos
        if obstacles[-1][0] < width - 300:
            obstacles.append([width, random.randint(50, height - 50 - obstacle_gap)])

        # 3. Eliminar obstáculos viejos
        if obstacles[0][0] < -obstacle_width:
            obstacles.pop(0)
            score += 1
            if score % 5 == 0: obstacle_speed += 2

        # 4. Detección de Colisiones
        # Hacemos la caja de colisión un poco más pequeña que el círculo visual para ser amables
        hitbox_margin = 10
        
        for obs in obstacles:
            obs_x, gap_y = obs
            # Si el jugador está horizontalmente dentro del muro...
            if (player_x + player_radius - hitbox_margin > obs_x and 
                player_x - player_radius + hitbox_margin < obs_x + obstacle_width):
                
                # Y verticalmente NO está en el hueco...
                if (player_y - player_radius + hitbox_margin < gap_y) or \
                   (player_y + player_radius - hitbox_margin > gap_y + obstacle_gap):
                    game_over = True

    # --- DIBUJADO ---
    
    # Obstáculos
    for obs in obstacles:
        obs_x, gap_y = obs
        cv2.rectangle(frame, (obs_x, 0), (obs_x + obstacle_width, gap_y), (0, 0, 255), -1)
        cv2.rectangle(frame, (obs_x, gap_y + obstacle_gap), (obs_x + obstacle_width, height), (0, 0, 255), -1)

    # Jugador
    color_player = (255, 0, 0) if not game_over else (0, 0, 150)
    cv2.circle(frame, (player_x, player_y), player_radius, color_player, -1)
    cv2.circle(frame, (player_x, player_y), player_radius, (255, 255, 255), 2)

    # Textos
    cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    if game_over:
        cv2.putText(frame, "GAME OVER", (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.putText(frame, "ESPACIO para reiniciar", (width//2 - 180, height//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Prototipo 3 - Face Jump (OpenCV)', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 27: # ESC
        break
    if key == 32: # ESPACIO
        reset_game()

cap.release()
cv2.destroyAllWindows()