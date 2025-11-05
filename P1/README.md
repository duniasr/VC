## Guion de prácticas de la asignatura Visión por Computador (VC)

Modesto Castrillón Santana  
Universidad de Las Palmas de Gran Canaria  
Escuela de Ingeniería en Informática  
Grado de Ingeniería Informática  
Curso 2025/2026 

** Dunia Suárez Rodríguez **


- # [Práctica 1](P1/README.md)
<!-- hola -->
## Descripción
En la Práctica 1 se realizan transformaciones básicas en imágenes y vídeos haciendo uso de las librerías; OpenCV, NumPy y Matplotlib. En específico con la práctica se pretende entender el funcionamiento de la captura de imágenes desde la webcam y el manejo de píxeles.

## Requisitos
Para esta práctica haremos uso de Python 3.11, Anaconda y de las librerías:
- OpenCV
- NumPy
- Matplotlib
Además nos conectaremos al entorno 'VC_P1' creado con Anaconda.

## Modo de ejecución
1. Crear el entorno de Anaconda(solo la primera vez):
    `conda create --name VC_P1 python=3.11.5`
1. Activar el entorno en Anaconda Prompt:
    `conda activate VC_P1`
2. Abrir el cuaderno:
    `practica1.ipynb`
    y seleccionar el kernel correspondiente `VC_P1` desde Select Kernel o desde CTRL+SHIFT+P -> Python: Selleccionar intérprete.
    Si es necesario instala ipykernel en el entorno:
    `conda install -n VC_P1 ipykernel --update-deps --force-reinstall`
3. Ejecutar el cuaderno secuencialmente

## Resultados
La visualización de los resultados se obtiene al ejecutar el código. Desglosamos ahora cada ejercicio:
La IA ChatGPT se utilizó como recurso de apoyo para aclarar dudas, explorar funcionalidades de la librería que no conocía y obtener orientación en el ejercicio 5.

### Ejercicio 1
Objetivo -> generar una imagen de 800x800 píxeles con el patrón de un tablero de ajedrez. Hay dos versiones:
- Manual: 
Primero en la línea `np.zeros((alto,ancho,1), dtype = np.uint8)`, creamos una imagen en negro de 800x800 píxeles con 1 canal(escala de grises: blanco y negro). Sobre esta imagen pintaremos los cuadros blancos y negros uno a uno con las líneas de código `tablero[alto,ancho,canal] = 255`-> alto:eje y; ancho: eje x; canal: 0 porque solo hay un canal y 255 porque pintamos el blanco.

- Con bucle:
Primero definimos el tamaño de cada casilla "100" pues en un tablero de ajedrez hay 8x8 casillas, luego generamos la matriz con `np.zeros((800, 800, 1)`, de nuevo un canal(escala de grises). A continuación recorremos la matriz saltando de 100 en 100 píxeles y pintamos de blanco si la suma de índices es par y de negro si es impar.


### Ejercicio 2
Objetivo -> crear una imagen al estilo Mondrian
Primero generamos una imagen con 3 planos con `np.zeros((alto,ancho,3)` y la pintamos de blanco `colorinchis[:,:,canal] = 255`. 

Seguidamente generamos los rectángulos usando la funcion de openCV `cv2.rectangle(imagen, (x1, y1), (x2, y2), (B, G, R), grosor)`, donde "(x1, y1) es la esquina superior izquierda del rectángulo y (x2, y2) la esquina inferior derecha". 

Finalmente usamos la función también de openCV `cv2.line(imagen, (x_inicio, y_inicio), (x_fin, y_fin), (B, G, R), grosor)` para generar las líneas y que queden por encima de los rectángulos ya creados.


### Ejercicio 3
Objetivo -> Modificar los valores de un plano de la imagen
Para modificar los planos se utilizó la inversión de uno de ellos(el verde) restando `nuevoValor = 255 - valorOg`.

También se puso que en cada imagen se muestre el color del canal, para ello se usó de la función `np.zeros_like(img)` para copiar el arreglo del frame ya creado y luego se activó solo el color del canal con `red_image[:, :, canal] = r/g/b`


### Ejercicio 4
Objetivo -> Detectar el píxel más claro y más oscuro y dibujar un círculo sobre ellos. Además hacer lo mismo en las zonas 8x8 más oscuras y claras.
Primero con las funciones `cv2.VideoCapture(0)` y `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` abrir la cámara web y convertir cada frame a escala de grises(para evaluar la intensidad) respectivamente. Con `vid.read()` obtenemos un booleano que confirma si se capturó imagen, y si lo hizo se obtiene también la imagen.

A continuación creamos variables para almacenar la suma más alta y más baja encontradas en un bloque y 2 variables para guardar las coordenadas (x, y) de la esquina superior izquierda del bloque más claro y del más oscuro.
Luego recorremos la imagen en bloques de 8, en dicho bloque hacemos una suma total de intensidades con la función `np.sum(bloque)`, los bloques con más blanco tendrán suma alta y los más oscuros suma baja. Comparamos cada suma con los valores anteriormente guardados.

Por último con `cv2.rectangle(imagen, esquina_sup_izq, esquina_inf_der, color, grosor)` dibujamos rectángulos alrededor de las zonas 8x8 calculadas y mostramos la imagen con `cv2.imshow(titulo, img)`.


### Ejercicio 5
Objetivo -> crear propuesta de Pop Art desde la imagen capturada en cámara.
Primero reducimos la resolución con `cv2.resize(frame, (w//ncells, h//ncells), cv2.INTER_NEAREST)`, dividiendo la imagen en una cuadrícula de celdas más pequeñas para trabajar en bloques de colores.

Se define una paleta de colores saturados Pop Art en formato BGR (amarillo, rojo, magenta, azul). Con la función `nearest_color()` calculamos la distancia euclidiana entre el color promedio de cada celda y los colores de la paleta, devolviendo siempre el color más parecido.

A continuación, para cada celda se obtiene la intensidad promedio con `np.mean(bloque)` y se calcula un radio de círculo proporcional a la luminosidad (zonas claras → círculos grandes, zonas oscuras → círculos pequeños). Este círculo se dibuja centrado en su celda con `cv2.circle()`, primero rellenando con el color de la paleta y luego añadiendo un contorno negro para darle un estilo cómic.

Finalmente, la imagen resultante se muestra en una ventana con `gitcv2.imshow('Pop Art', comic_frame)`.


<!-- - [Práctica 2](P2/README.md) -->
<!-- - [Práctica 3](P3/README.md) -->
<!-- - [Práctica 4](P4/README.md) -->
<!-- - [Práctica 5](P5/README.md) -->
<!-- - [Práctica 6](P6/README.md) -->
<!-- - [Práctica 7](P7/README.md) -->
<!-- - [Trabajo](Trabajo/README.md) -->
***
Obra bajo licencia de Creative Commons Reconocimiento - No Comercial 4.0 Internacional
