import cv2
import joblib
import numpy as np
from deepface import DeepFace

# ---------- CONFIGIGURACIÓN DEL FILTRO DE EMOCIONES ----------
# Umbral de confianza mínimo para considerar válida una predicción de emoción.
# Si la probabilidad es menor a 0.55 (55%), se descarta o se ignora.
CONF_THRESHOLD = 0.55
# Tiempo de espera entre cambios de filtro (en segundos).
COOLDOWN_SEC = 0.0

# Diccionario de estilos de reacción para cada emoción detectada.
# Define el color (tinte) y la opacidad (alpha) que se aplicará a la imagen.
# Formato de color: (B, G, R) - Azul, Verde, Rojo
REACTION_STYLE = {
    "happy":    {"tint": (0, 255, 255),   "alpha": 0.35},   # amarillo
    "fearful":  {"tint": (200, 100, 255), "alpha": 0.35},   # lila
    "sad":      {"tint": (255, 0, 0),     "alpha": 0.35},   # azul
    "disgusted":{"tint": (0, 255, 0),     "alpha": 0.35},   # verde
    "neutral":  {"tint": (0, 165, 255),   "alpha": 0.30},   # naranja
    "angry":    {"tint": (0, 0, 255),     "alpha": 0.35},   # rojo
    "surprised":{"tint": (255, 0, 255),   "alpha": 0.30},   # rosa
}

# ---------- FUNCIÓN AUXILIAR ----------
def apply_tint(frame, color_bgr, alpha):
    """
    Aplica un filtro de color (tinte) sobre toda la imagen utilizando mezcla ponderada.
    """
    # Si no hay color o la opacidad es 0, se retorna la imagen original
    if color_bgr is None or alpha <= 0:
        return frame
    # Crea una imagen "overlay" del mismo tamaño que el frame original, con el color seleccionado
    overlay = np.full_like(frame, color_bgr, dtype=np.uint8)
    # Suma ponderada para la mezcla entre la imagen original y el overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

# ---------- CARGAR MODELOS ----------
# Carga el modelo SVM (Support Vector Machine)
svm_model = joblib.load('emotion_svm_model.pkl')
# Carga el escalador (MinMaxScaler) usado durante el entrenamiento.
# Es crucial usar el MISMO escalador para normalizar los nuevos datos de entrada de la misma manera.
scaler = joblib.load('emotion_scaler.pkl')
# Carga la lista con los nombres de las clases
class_names = joblib.load('emotion_class_names.pkl')
labels = list(class_names)

# ---------- CÁMARA ----------
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
print("Cámara iniciada. ESC para salir.")

while True:
    # Captura frame a frame desde la cámara.
    ret, frame = cap.read()
    if not ret:
        break

    # --- 1. Detección de Rostros ---
    try:
        # Utiliza DeepFace para localizar rostros en el frame actual.
        # img_path: La imagen a analizar.
        # detector_backend="mtcnn": MTCNN es un detector más preciso (aunque más lento) que Haar Cascades.
        # enforce_detection=False: Evita que el programa lance una excepción si no encuentra ninguna cara.
        faces = DeepFace.extract_faces(img_path=frame, detector_backend="mtcnn", enforce_detection=False)
    except:
        faces = []

    # Itera sobre cada cara detectada en la imagen.
    for face in faces:
        # --- 2. Filtrado por Confianza de Detección ---
        # Ignora las detecciones con baja probabilidad de ser una cara real (< 75%)
        if face.get('confidence', 0) < 0.75: 
            continue
        
        # Extrae las coordenadas del rectángulo delimitador de la cara
        fa = face['facial_area']
        x, y, w, h = int(fa['x']), int(fa['y']), int(fa['w']), int(fa['h'])
        # Asegura que las coordenadas estén dentro de los límites de la imagen 
        # para evitar errores de índice
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        # Crea un recorte (crop) de la región de la cara
        face_img = frame[y1:y2, x1:x2].copy()
        if face_img.size == 0:
            continue

        # --- 3. Extracción de Embeddings (Características Faciales) ---
        # Convierte el recorte de la cara de BGR a RGB (esperado por DeepFace/FaceNet)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        try:
            # Utiliza DeepFace para obtener el vector de características (embedding) de la cara
            emb = np.array(
                DeepFace.represent(face_rgb, model_name="Facenet", enforce_detection=False)[0]['embedding']
            ).reshape(1, -1)
        except:
            continue

        # --- 4. Clasificación de la Emoción ---
        emb_scaled = scaler.transform(emb)

        # Realiza la predicción usando el modelo SVM
        if hasattr(svm_model, "predict_proba"):
            # Si el modelo soporta probabilidades, obtenemos la confianza de cada clase
            probs = svm_model.predict_proba(emb_scaled)[0]
            idx = int(np.argmax(probs))     # Índice de la clase con mayor probabilidad
            prob = float(probs[idx])        # Valor de la probabilidad máxima
            label = labels[idx]             # Nombre de la emoción predicha
            accepted = (prob >= CONF_THRESHOLD) # Verifica si supera el umbral de confianza
        else:
            # Si no soporta probabilidades, obtenemos directamente la clase
            idx = int(svm_model.predict(emb_scaled)[0])
            label = labels[idx]
            prob = None
            accepted = True

        # --- 5. Visualización de Resultados ---
        # Dibuja un rectángulo alrededor de la cara detectada
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
        # Prepara el texto con la etiqueta y la probabilidad
        text = label if prob is None else f"{label} ({prob:.2f})"
        # Escribe el texto sobre el rectángulo de la cara
        cv2.putText(frame, text, (x1, y1 - 10), font, 0.8, (180, 180, 180), 1)

        # --- 6. Aplicación del Filtro de Tinte ---
        # Obtiene el estilo (color y alpha) correspondiente a la emoción detectada
        style = REACTION_STYLE.get(label, {})
        tint = style.get("tint")
        alpha = style.get("alpha", 0.0)
        # Si hay un tinte definido y opacidad > 0, aplica el filtro a todo el frame
        if tint is not None and alpha > 0:
            frame = apply_tint(frame, tint, alpha)

    cv2.imshow("Detector Emociones", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera la cámara y cierra todas las ventanas de OpenCV al salir
cap.release()
cv2.destroyAllWindows()
