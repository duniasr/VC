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
En esta fase procesamos el dataset de imágenes utilizando el modelo pre-entrenado FaceNet. Transformamos cada rostro en un vector numérico de 128 dimensiones (embedding) y almacenamos los resultados en disco para optimizar el tiempo de cómputo en fases posteriores

## Entrenamiento y optimización del clasificador SVM
Cargamos los embeddings generados previamente y aplicamos un preprocesamiento de normalización (MinMaxScaler). Posteriormente, entrenamos un clasificador de Máquina de Vectores de Soporte (SVM) utilizando GridSearchCV para encontrar automáticamente la combinación óptima de hiperparámetros (kernel, C, gamma) mediante validación cruzada.

## Prototipo Filtro 1(con modelo entrenado)

## Prototipo Filtro 2(libre)


<!-- - [Práctica 6](P6/README.md) -->
<!-- - [Práctica 7](P7/README.md) -->
<!-- - [Trabajo](Trabajo/README.md) -->
***
Obra bajo licencia de Creative Commons Reconocimiento - No Comercial 4.0 Internacional
