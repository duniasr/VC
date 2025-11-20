## Guion de prácticas de la asignatura Visión por Computador (VC)

Modesto Castrillón Santana  
Universidad de Las Palmas de Gran Canaria  
Escuela de Ingeniería en Informática  
Grado de Ingeniería Informática  
Curso 2025/2026 

** Dunia Suárez Rodríguez **


- # [Práctica 5]
## Descripción
En la Práctica 5 se realiza el entrenamiento de un modelo propio para la extracción de información biométrica (emociones), aplicando técnicas de Transfer Learning. Este modelo se integra posteriormente en una aplicación en tiempo real (filtro). Además, se incluye un segundo prototipo de filtro creativo de temática libre.

## Requisitos
Para esta práctica haremos uso de Python 3.11, Anaconda y de las librerías:
- opencv
- numpy
- time
- sklearn
- joblib
- deepface

Además nos conectaremos al entorno 'VC_P5' creado con Anaconda.

## Modo de ejecución
1.  *Crear el entorno (basado en P5):*
    conda create --name VC_P5 python=3.11.5
    conda activate VC_P4
2.  *Instalar librerías principales:*
    pip install opencv-python numpy scikit-learn joblib deepface
3. *Ejecutar el código de "Extracción de características".*
4. *Ejecutar el código de "Entrenamiento".*
5. *Ejecutar el script "prototipo_emociones_animado.py" y "prototipo_juegoVJ" para la visualización de los 2 filtros.*


## Resultados
La IA Gemini se utilizó como recurso de apoyo para aclarar dudas, explorar funcionalidades de la librerías que no conocía y obtener orientación.

## Extracción de características(Embeddings)
En esta primera fase, el objetivo es transformar las imágenes "crudas" (píxeles) en una representación matemática compacta que un modelo de Machine Learning pueda procesar. No se realiza entrenamiento aquí, solo procesamiento de datos.
El dataset original que he escogido tiene cerca de 4000 imagenes por clase, así que he limitado el número de imagenes por clase `MAX_IMAGES_PER_CLASS = 600`.

Utilizamos el modelo FaceNet (a través de DeepFace), que requiere una entrada de imágenes de 160x160 píxeles.
* Es una Red Neuronal Convolucional (CNN) desarrollada por investigadores de Google. Su arquitectura está diseñada específicamente para transformar imágenes de rostros, ignorando detalles irrelevantes (iluminación, pose) y centrarse en los rasgos biométricos únicos, condensándolos en un vector de 128 dimensiones.

La función principal es `LoadDataset` que recorre el directorio de imágenes, procesa cada cara con DeepFace para obtener su "huella digital" numérica (embedding) y devuelve los arrays X (características) e Y (etiquetas) necesarios para entrenar el modelo SVM. Para ello realiza lo siguiente en cada imagen:
- Carga la imagen con `cv2.imread` y la redimensiona a las dimensiones requeridas por el modelo(160x160).
- Con la función `DeepFace.represent` se pasa la imagen por la red neuronal pre-entrenada 'Facenet' y la convierte en un vector de 128 números (el embedding). El parámetro `enforce_detection=False` evita que el script se detenga si DeepFace no encuentra una cara clara en una foto específica. Tras esta función se extrae el vector `img_embedding` de 128 números que representa las características faciales únicas de esa imagen.
- Los datos se guardan en 
    * X(Features) -> matriz donde cada fila es el vector de 128 números de una cara
    * Y(Labels) -> vector con el número entero correspondiente a la emoción (ej: 0 para 'angry', 1 para 'happy').

Una vez ejecutada la función que recorre las imágenes, convertimos las listas X e Y a numpy arrays(necesario para Scikit-Learn).

Antes de procesar nada, construimos el modelo FaceNet en memoria. Esto es necesario para obtener dinámicamente las dimensiones de entrada que requiere la red. Pasamos estas dimensiones (dim) a la función de carga para asegurar que todas las imágenes se redimensionen al tamaño exacto. 

Continuamos con la llamada la función `LoadDataset`. Esta devuelve los arrays X (datos) e Y (etiquetas), así como estadísticas sobre el número de muestras procesadas. 

Para finalizar, usamos la librería joblib para serializar y guardar las matrices X e Y en archivos .pkl (embeddings_X.pkl, labels_Y.pkl). Esto permite que el siguiente script (entrenamiento) cargue los datos instantáneamente sin tener que volver a procesar las imágenes.
Se guardan tres archivos en disco:
* `embeddings_X.pkl`: La matriz matemática con todas las caras procesadas.
* `labels_Y.pkl`: Las respuestas correctas (qué emoción es cada vector).
* `emotion_class_names.pkl`: Nos permitirá traducir en el futuro que la etiqueta 0 significa "angry" o la 3 significa "happy".

## Entrenamiento y optimización del clasificador SVM
Una vez convertidas las imágenes a vectores numéricos, utilizamos un algoritmo de aprendizaje supervisado para encontrar patrones que diferencien una emoción de otra.

Primero cargamos los arrays X e Y generados en el paso anterior normalizaciamos sus valores con `MinMaxScaler`. Se normalizan pues los algoritmos basados en distancias como el SVM son muy sensibles a la escala de los datos. Si no normalizamos, una característica con valores numéricos altos dominaría la decisión sobre otras con valores pequeños, degradando el rendimiento.

Continuamos con la configuración del clasificador (SVM con Kernel RBF). 
* Utilizamos una Máquina de Vectores de Soporte (SVC).
    SVC (Support Vector Classification) es un algoritmo de aprendizaje supervisado que busca encontrar el hiperplano óptimo (una frontera de decisión) que separe las diferentes clases de datos con el mayor margen posible. Se usa SVC pues es altamente efectivo trabajando con vectores de alta dimensión (como los embeddings de 128 valores de FaceNet).
* Elegimos el kernel `rbf` (Radial Basis Function). A diferencia de un kernel lineal, el RBF permite trazar fronteras de decisión curvas y complejas, lo cual es fundamental para datos biométricos donde las diferencias entre "feliz" y "neutro" no son linealmente separables.
* Usamos `class_weight='balanced` para que el modelo preste más atención a las clases con menos fotos, evitando que se sesgue hacia la emoción mayoritaria.

`GridSearchCV` es usado para sistematizar el entrenamiento. En lugar de entrenar un único modelo con parámetros aleatorios, GridSearchCV genera múltiples versiones del clasificador SVM combinando diferentes valores de regularización (C) y coeficientes de kernel (Gamma), seleccionando finalmente el modelo que ofrece la mayor precisión estadística. Los parámetros críticos son:
* C (Regularización): Controla la rigidez del margen. Un valor alto intenta clasificar todo correctamente (riesgo de sobreajuste), mientras que uno bajo permite más errores buscando una frontera más suave.
* Gamma: Define el alcance de la influencia de un solo ejemplo de entrenamiento.
A continuación, el método `.fit()` ejecuta el proceso de búsqueda y entrenamiento definido por `GridSearchCV`.

Finalmente, guardamos los dos archivos esenciales para el prototipo:
* `emotion_svm_model.pkl`: El clasificador SVM ya entrenado.
* `emotion_scaler.pkl`: La "regla" de normalización. Es vital guardar el scaler para aplicarle exactamente la misma transformación matemática a las caras que capte la webcam en el futuro.


## Prototipo Filtro 1(con modelo entrenado)
Este script integra el modelo entrenado en una aplicación de visión artificial que captura vídeo, procesa rostros frame a frame y genera una respuesta visual (pantalla de color) basada en la emoción detectada.

Comenzamos con la carga de los tres archivos esenciales generados anteriormente:
* `emotion_svm_model.pkl`
* `emotion_scaler.pkl`
* `emotion_class_names.pkl`

Continuamos con la captura y detección facial (MTCNN). Mediante un bucle analizamos cada fotograma y utilizamos DeepFace.extract_faces con el backend MTCNN (Multi-task Cascaded Convolutional Networks). Usamos MTCNN, en vez de otro detector, pues es mucho más robusto ante variaciones de luz y pose, lo que mejora la experiencia de usuario en tiempo real, aunque requiere más cómputo. Aplicamos un filtro de confianza (> 0.75) para descartar falsos positivos.

Para cada rostro validado:
* Recortamos la región de interés de la cara y la convertimos a RGB y con `DeepFace.represent` (modelo FaceNet) convertimos la cara en el vector de 128 características. 
* Normalizamos los valores del vector con `scaler.transform`. 
* Utilizamos `svm_model.predict_proba` que nos da la clase ganadora y el porcentaje de certeza.
* Si la certeza del modelo supera el 55%(`CONF_THRESHOLD`), se activa la reacción.
* Implementamos una función `apply_tint` que utiliza `cv2.addWeighted` para fusionar la imagen original con una capa de color semitransparente.

Cada emoción tiene asignado un código de color (ej: Amarillo para "happy", Verde para "disgusted"), proporcionando feedback visual instantáneo y aumentado sobre la realidad.

## Prototipo Filtro 2(libre)


<!-- - [Práctica 6](P6/README.md) -->
<!-- - [Práctica 7](P7/README.md) -->
<!-- - [Trabajo](Trabajo/README.md) -->
***
Obra bajo licencia de Creative Commons Reconocimiento - No Comercial 4.0 Internacional
