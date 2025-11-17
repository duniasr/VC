import cv2
import joblib
import numpy as np
import os
import time
from deepface import DeepFace

# ---------- CONFIG SOLO TINTE ----------
CONF_THRESHOLD = 0.55
COOLDOWN_SEC = 0.0   # totalmente instantáneo

REACTION_STYLE = {
    "happy":    {"tint": (0, 255, 255),   "alpha": 0.35},   # amarillo
    "fearful":  {"tint": (200, 100, 255), "alpha": 0.35},   # lila
    "sad":      {"tint": (255, 0, 0),     "alpha": 0.35},   # azul
    "disgusted":{"tint": (0, 255, 0),     "alpha": 0.35},   # verde
    "neutral":  {"tint": (0, 165, 255),   "alpha": 0.30},   # naranja
    "angry":    {"tint": (0, 0, 255),     "alpha": 0.35},   # rojo
    "surprised":{"tint": (255, 0, 255),   "alpha": 0.30},   # rosa
}

def apply_tint(frame, color_bgr, alpha):
    if color_bgr is None or alpha <= 0:
        return frame
    overlay = np.full_like(frame, color_bgr, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

# ---------- Cargar modelos ----------
svm_model = joblib.load('emotion_svm_model.pkl')
scaler = joblib.load('emotion_scaler.pkl')
class_names = joblib.load('emotion_class_names.pkl')
labels = list(class_names)

# ---------- Cámara ----------
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
print("Cámara iniciada. ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        faces = DeepFace.extract_faces(img_path=frame, detector_backend="mtcnn", enforce_detection=False)
    except:
        faces = []

    for face in faces:
        if face.get('confidence', 0) < 0.75:
            continue

        fa = face['facial_area']
        x, y, w, h = int(fa['x']), int(fa['y']), int(fa['w']), int(fa['h'])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

        face_img = frame[y1:y2, x1:x2].copy()
        if face_img.size == 0:
            continue

        # Embedding
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        try:
            emb = np.array(
                DeepFace.represent(face_rgb, model_name="Facenet", enforce_detection=False)[0]['embedding']
            ).reshape(1, -1)
        except:
            continue

        # Clasificar emoción
        emb_scaled = scaler.transform(emb)

        if hasattr(svm_model, "predict_proba"):
            probs = svm_model.predict_proba(emb_scaled)[0]
            idx = int(np.argmax(probs))
            prob = float(probs[idx])
            label = labels[idx]
            accepted = (prob >= CONF_THRESHOLD)
        else:
            idx = int(svm_model.predict(emb_scaled)[0])
            label = labels[idx]
            prob = None
            accepted = True

        # Dibujo básico
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
        text = label if prob is None else f"{label} ({prob:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10), font, 0.8, (180, 180, 180), 1)

        # === SOLO TINTE AQUÍ ===
        style = REACTION_STYLE.get(label, {})
        tint = style.get("tint")
        alpha = style.get("alpha", 0.0)
        if tint is not None and alpha > 0:
            frame = apply_tint(frame, tint, alpha)

    cv2.imshow("Detector Emociones - SOLO TINTE", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
