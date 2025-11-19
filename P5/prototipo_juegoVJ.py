import cv2
import numpy as np
import random
import os 

print("Iniciando Prototipo 3: Face Jump (¡Estética Mejorada!) 🎮")

# --- 1. Configuración ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
width, height = 640, 480

# --- 2. Carga de Assets ---
def load_asset(path, resize_to=None):
    if not os.path.exists(path):
        print(f"ERROR: Falta {path}")
        exit()
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if resize_to:
        img = cv2.resize(img, resize_to)
    return img

def overlay_png(background, overlay, x, y):
    h_bg, w_bg = background.shape[:2]
    h_ov, w_ov = overlay.shape[:2]

    if x >= w_bg or y >= h_bg or x + w_ov <= 0 or y + h_ov <= 0:
        return background

    bg_x, bg_y = max(0, x), max(0, y)
    ov_x, ov_y = max(0, -x), max(0, -y)
    
    h_part = min(h_bg - bg_y, h_ov - ov_y)
    w_part = min(w_bg - bg_x, w_ov - ov_x)

    if h_part <= 0 or w_part <= 0: return background

    roi_bg = background[bg_y:bg_y+h_part, bg_x:bg_x+w_part]
    roi_ov = overlay[ov_y:ov_y+h_part, ov_x:ov_x+w_part]

    if roi_ov.shape[2] == 4:
        alpha = roi_ov[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(0, 3):
            roi_bg[:, :, c] = (alpha * roi_ov[:, :, c] + alpha_inv * roi_bg[:, :, c])
        background[bg_y:bg_y+h_part, bg_x:bg_x+w_part] = roi_bg
    else:
        background[bg_y:bg_y+h_part, bg_x:bg_x+w_part] = roi_ov

    return background

# Cargar imágenes
player_img = load_asset("player.png", resize_to=(90, 90)) 
player_size = player_img.shape[0]
wall_top_img = load_asset("wall_top.png")
wall_bottom_img = load_asset("wall_bottom.png")
background_img = load_asset("background.png", resize_to=(width, height))

# --- 3. Variables ---
player_x = 100          
player_radius = player_size // 2 
obstacle_width = 60 
obstacle_speed = 7 
min_obstacle_gap = 180
max_obstacle_gap = 280 
score = 0
game_over = False
obstacles = [] 

def create_new_obstacle():
    gap_size = random.randint(min_obstacle_gap, max_obstacle_gap)
    gap_y_start = random.randint(50, height - 50 - gap_size)
    return [width, gap_y_start, gap_size]

def reset_game():
    global obstacles, score, game_over, obstacle_speed
    obstacles = []
    score = 0
    obstacle_speed = 7 
    game_over = False
    obstacles.append(create_new_obstacle())

reset_game()

# --- 4. Bucle ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    
    # --- PASO 1: DETECTAR LA CARA (USANDO LA IMAGEN DE LA CÁMARA) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Parametros ajustados para ser más sensible (1.1, 4)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    player_y = height // 2 
    face_detected = False

    if len(faces) > 0:
        (x, y, w, h) = faces[0] 
        player_y = int(y + h / 2)
        face_detected = True
    
    # --- PASO 2: DIBUJAR EL JUEGO (AHORA SÍ TAPAMOS LA CÁMARA) ---
    # Sobrescribimos el frame de la cámara con el fondo del espacio
    frame[:] = background_img[:] 

    # --- Lógica ---
    if not game_over:
        for obs in obstacles:
            obs[0] -= obstacle_speed
        
        if obstacles[-1][0] < width - 250: 
            obstacles.append(create_new_obstacle())

        if obstacles[0][0] < -obstacle_width:
            obstacles.pop(0)
            score += 1
            if score % 5 == 0: obstacle_speed += 1 

        for obs in obstacles:
            obs_x, gap_y, current_gap_size = obs
            margin = 15 
            if (player_x + player_radius - margin > obs_x and 
                player_x - player_radius + margin < obs_x + obstacle_width):
                if (player_y - player_radius + margin < gap_y) or \
                   (player_y + player_radius - margin > gap_y + current_gap_size):
                    game_over = True

    # --- Dibujado de Elementos ---
    for obs in obstacles:
        obs_x, gap_y, current_gap_size = obs
        
        # Muro superior
        top_h = gap_y
        if top_h > 0:
            top_wall = cv2.resize(wall_top_img, (obstacle_width, top_h))
            frame = overlay_png(frame, top_wall, obs_x, 0)

        # Muro inferior
        bot_h = height - (gap_y + current_gap_size)
        if bot_h > 0:
            bot_wall = cv2.resize(wall_bottom_img, (obstacle_width, bot_h))
            frame = overlay_png(frame, bot_wall, obs_x, gap_y + current_gap_size)

    # Jugador
    frame = overlay_png(frame, player_img, player_x - player_radius, player_y - player_radius)

    # UI
    cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    if not face_detected:
        # Aviso si no te ve
        cv2.putText(frame, "NO TE VEO!", (width//2 - 100, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if game_over:
        cv2.putText(frame, "GAME OVER", (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.putText(frame, "ESPACIO para reiniciar", (width//2 - 180, height//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Prototipo 3 - Face Jump (Final)', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 27: break
    if key == 32: reset_game()

cap.release()
cv2.destroyAllWindows()