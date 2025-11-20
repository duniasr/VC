import cv2
import numpy as np
import random
import os 

print("Iniciando Prototipo")

# --- CONFIGURACIÓN ---
# Carga el clasificador Haar Cascade para detección de rostros frontal.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
width, height = 640, 480 # Dimensión ventana del juego

# --- CARGA Y GESTIÓN DE ASSETS ---
def load_asset(path, resize_to=None):
    """
    Carga una imagen desde disco, verificando que exista.
    Mantiene el canal Alfa (transparencia) si la imagen es PNG.
    """
    if not os.path.exists(path):
        print(f"ERROR: Falta {path}")
        exit()
    # IMREAD_UNCHANGED es vital para cargar la transparencia (4 canales: BGRA)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if resize_to:
        img = cv2.resize(img, resize_to)
    return img

def overlay_png(background, overlay, x, y):
    """
    Superpone una imagen PNG con transparencia sobre un fondo opaco.
    Maneja los bordes de la pantalla para evitar errores si el objeto sale del marco.
    """
    # Obtiene las dimensiones (alto y ancho) de la imagen de fondo y del objeto
    h_bg, w_bg = background.shape[:2]
    h_ov, w_ov = overlay.shape[:2]

    # Si la imagen está totalmente fuera de la pantalla, no hacemos nada
    if x >= w_bg or y >= h_bg or x + w_ov <= 0 or y + h_ov <= 0:
        return background

    # Coordenadas de inicio en el FONDO
    bg_x, bg_y = max(0, x), max(0, y)
    # Coordenadas de inicio en el OBJETO
    ov_x, ov_y = max(0, -x), max(0, -y)
    # Calculamos el tamaño del área rectangular que vamos a pintar (la intersección visible)
    # Es el mínimo entre "lo que queda de fondo" y "lo que queda de objeto"
    h_part = min(h_bg - bg_y, h_ov - ov_y)
    w_part = min(w_bg - bg_x, w_ov - ov_x)

    if h_part <= 0 or w_part <= 0: return background
    # Recortamos los trozos exactos de ambas imágenes que se van a mezclar
    roi_bg = background[bg_y:bg_y+h_part, bg_x:bg_x+w_part]
    roi_ov = overlay[ov_y:ov_y+h_part, ov_x:ov_x+w_part]
    # Verificamos si el objeto tiene 4 canales (B, G, R, Alpha)
    if roi_ov.shape[2] == 4:
        # Normalizamos el canal Alfa de 0-255 a 0.0-1.0 para poder multiplicar
        alpha = roi_ov[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha # La parte inversa (lo que se verá del fondo)
        # Mezclamos cada canal de color (Azul, Verde, Rojo) por separado
        for c in range(0, 3):
            # Fórmula: (Color Objeto * Transparencia) + (Color Fondo * (1 - Transparencia))
            roi_bg[:, :, c] = (alpha * roi_ov[:, :, c] + alpha_inv * roi_bg[:, :, c])
        # Ponemos el trozo mezclado de vuelta en la imagen de fondo original
        background[bg_y:bg_y+h_part, bg_x:bg_x+w_part] = roi_bg
    else:
        background[bg_y:bg_y+h_part, bg_x:bg_x+w_part] = roi_ov

    return background

# Cargar imágenes
player_img = load_asset("player.png", resize_to=(90, 90)) # Alien
player_size = player_img.shape[0]
wall_top_img = load_asset("wall_top.png") # Columnas superiores
wall_bottom_img = load_asset("wall_bottom.png") # Columnas inferiores
background_img = load_asset("background.png", resize_to=(width, height)) # Fondo galazia

# --- LÓGICA DEL JUEGO ---
# Estado inicial del jugador y obstáculos
player_x = 100              # Posición horizontal fija del jugador(100pxs desde la izq)
player_radius = player_size // 2 
obstacle_width = 60         # Ancho de los muros
obstacle_speed = 7          # Velocidad inicial
min_obstacle_gap = 180      # Hueco mínimo entre muro de arriba y abajo
max_obstacle_gap = 280      # Hueco máximo entre muro de arriba y abajo
score = 0
game_over = False
obstacles = [] 

def create_new_obstacle():
    """Genera un obstáculo nuevo fuera de la pantalla por la derecha"""
    gap_size = random.randint(min_obstacle_gap, max_obstacle_gap)
    # El hueco empieza en una altura aleatoria, respetando márgenes
    gap_y_start = random.randint(50, height - 50 - gap_size)
    return [width, gap_y_start, gap_size]

def reset_game():
    """Reinicia todas las variables para empezar de cero"""
    global obstacles, score, game_over, obstacle_speed
    obstacles = []
    score = 0
    obstacle_speed = 7 
    game_over = False
    obstacles.append(create_new_obstacle())

# Iniciamos el juego
reset_game()

# --- BUCLE ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Espejo (flip) para que moverse a la izquierda sea intuitivo, y resize
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    
    # --- 1. Detección facial ---
    # Usamos la imagen de la cámara para detectar dónde está la cabeza del jugador
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detectamos caras. Parámetros ajustados para rapidez y estabilidad.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    # Posición vertical por defecto (centro)
    player_y = height // 2 
    face_detected = False

    # Si hay cara, la posición Y del jugador será el centro de la cara detectada
    if len(faces) > 0:
        (x, y, w, h) = faces[0] 
        player_y = int(y + h / 2)
        face_detected = True
    
    # --- 2. Dibujar el juego ---
    # Tapamos la imagen de la cámara con el fondo del juego
    frame[:] = background_img[:] 

    # --- Lógica de Movimiento y Colisión ---
    if not game_over:
        # Mover obstáculos hacia la izquierda
        for obs in obstacles:
            obs[0] -= obstacle_speed
        
        # Generar nuevos obstáculos si el último ya avanzó suficiente
        if obstacles[-1][0] < width - 250: 
            obstacles.append(create_new_obstacle())

        # Eliminar obstáculos que salieron de la pantalla y sumar puntos
        if obstacles[0][0] < -obstacle_width:
            obstacles.pop(0)
            score += 1
            # Aumentar velocidad progresivamente
            if score % 5 == 0: obstacle_speed += 3 

        # Detección de Colisiones
        for obs in obstacles:
            obs_x, gap_y, current_gap_size = obs
            # Margen de gracia para facilitar el juego 
            margin = 15
            # A) COMPROBACIÓN HORIZONTAL (¿Estoy cruzando el muro?)
            # Si mi lado derecho es mayor que el inicio del muro...
            # Y mi lado izquierdo es menor que el final del muro...
            if (player_x + player_radius - margin > obs_x and 
                player_x - player_radius + margin < obs_x + obstacle_width):
                # B) COMPROBACIÓN VERTICAL (¿Estoy chocando verticalmente?)
                if (player_y - player_radius + margin < gap_y) or \
                   (player_y + player_radius - margin > gap_y + current_gap_size):
                    game_over = True

    # --- Dibujar Elementos, superponer imágenes ---
    for obs in obstacles:
        obs_x, gap_y, current_gap_size = obs
        
        # Muro superior
        top_h = gap_y
        if top_h > 0:
            # Redimensionamos la textura del muro para que encaje en la altura necesaria
            top_wall = cv2.resize(wall_top_img, (obstacle_width, top_h))
            frame = overlay_png(frame, top_wall, obs_x, 0)

        # Muro inferior
        bot_h = height - (gap_y + current_gap_size)
        if bot_h > 0:
            # Redimensionamos la textura del muro para que encaje en la altura necesaria
            bot_wall = cv2.resize(wall_bottom_img, (obstacle_width, bot_h))
            frame = overlay_png(frame, bot_wall, obs_x, gap_y + current_gap_size)

    # Jugador alien
    # Se dibuja en la posición fija X y la posición Y controlada por la cara
    frame = overlay_png(frame, player_img, player_x - player_radius, player_y - player_radius)

    # UI
    cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Feedback si se pierde el tracking facial
    if not face_detected:
        cv2.putText(frame, "NO TE VEO!", (width//2 - 100, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Pantalla de Game Over
    if game_over:
        cv2.putText(frame, "GAME OVER", (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.putText(frame, "ESPACIO para reiniciar", (width//2 - 180, height//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Prototipo 3 - Face Jump (Final)', frame)

    # Controles de teclado
    key = cv2.waitKey(10) & 0xFF
    if key == 27: break        # ESC para salir
    if key == 32: reset_game() # ESPACIO para reiniciar

# Libera la cámara y cierra todas las ventanas de OpenCV al salir
cap.release()
cv2.destroyAllWindows()