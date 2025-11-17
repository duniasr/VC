import cv2
import joblib
import numpy as np
from deepface import DeepFace

# Cargar modelos
svm_model = joblib.load('emotion_svm_model.pkl')
scaler = joblib.load('emotion_scaler.pkl')
class_names = joblib.load('emotion_class_names.pkl')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
color_reaccion = (0, 255, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = DeepFace.extract_faces(img_path=frame, detector_backend="mtcnn", enforce_detection=False)

    for face in faces:
        if face['confidence'] < 0.9:
            continue

        x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        embedding_obj = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
        embedding = np.array(embedding_obj[0]['embedding']).reshape(1, -1)
        embedding_scaled = scaler.transform(embedding)
        prediction_index = int(svm_model.predict(embedding_scaled)[0])
        emotion_label = class_names[prediction_index]

        # Dibujar
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_reaccion, 2)
        cv2.putText(frame, emotion_label, (x, y-10), font, 0.9, color_reaccion, 2, cv2.LINE_AA)

    # Mostrar ventana nativa
    cv2.imshow('Detector de Emociones', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
