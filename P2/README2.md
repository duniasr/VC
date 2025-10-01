## Guion de prácticas de la asignatura Visión por Computador (VC)  
Modesto Castrillón Santana  
Universidad de Las Palmas de Gran Canaria  
Escuela de Ingeniería en Informática  
Grado de Ingeniería Informática  
Curso 2025/2026  

**Dunia Suárez Rodríguez**

<!-- - [Práctica 2](P2/README.md) -->

## Descripción
En la Práctica 1 se realizan transformaciones básicas en imágenes y vídeos haciendo uso de las librerías; OpenCV, NumPy , Matplotlib y Pillow. En específico con la práctica se pretende entender el funcionamiento para la deteccion de bordes y la sustracción de fondo.

## Requisitos
Para esta práctica haremos uso de Python 3.11, Anaconda y de las librerías:  
- OpenCV  
- NumPy  
- Matplotlib  
- Y los módulos de Pillow(PIL) siguientes:  
    - Image  
    - ImageDraw  
    - ImageFont  

Además nos conectaremos al entorno 'VC_P1' creado con Anaconda.

## Resultados
La visualización de los resultados se obtiene al ejecutar el código. Desglosamos ahora cada ejercicio:  
El uso de la IA ChatGPT fue usado para: resolver dudas, aplicar funcionalidades y funciones de la librería que desconocía y guiarme en el ejercicio 4

### Ejercicio 1
Objetivo -> realizar la cuenta de píxeles blancos por filas y determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil.

Empezamos aplicando la función `cv2.Canny` con la que obtenemos los contornos detectados(valor 255) y el resto(valor 0), seguida de la función `cv2.reduce` para sumar los valores de los píxeles a lo largo de un eje(en este caso del eje y).  

Luego obtenemos el número de píxeles blancos por fila. En dichas filas aplicaremos la función `max()` para calcular el máximo número de píxeles blancos en una fila, valor que usaremos para calcular el umbral(90% del valor máximo).  
Seguidamente comienza la iteración de las filas en busca de aquíellas que superen o igualen el umbral definido, así pues las filas que lo cumplan serán impresas por texto y marcadas en la gráfica.

Al comparar, observamos que usando sobel, en la imagen se resaltan más los bordes horizontales fuertes como los ojos y frente, donde el motivo de tantas líneas puede deberse a algo de ruido. Por la otra parte, en la imagen que usa canny, se captan más los bordes de las estructuras verticales, como la nariz.

### Ejercicio 2
Objetivo -> aplicar umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realizar el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo.

Primero determinamos un umbral que usaremos con las funciones `cv2.THRESH_BINARY` y con `cv2.threshold` para convertir los gradientes Sobel en una imagen binaria, dejando solo los bordes más significativos(si superan el umbral se pone a 255 el píxel, si no se pone a 0).

Luego, usando Sobel, contamos el nº de píxeles por filas y columnas con la función `np.count_nonzero`, calculamos el valor máximo en dichas filas y columnas `max()` y después creamos listas que contienen las filas y columnas que cumplen el umbral usando la función `np.where`.  
Hacemos el mismo procedimiento para el método Canny. Para visualizar estos resultados dibujamos líneas en las filas y columnas significativas con `cv2.line`.

Finalmente mostramos ambas imágenes Sobel y Canny.

### Ejercicio 3
Objetivo ->  proponer un demostrador que capture las imágenes de la cámara, y les permita exhibir lo aprendido

En esta tarea hemos creado el demostrador con 3 modos a parte del la cámara original.

El primer modo es un filtro de color que a medida que cicla entre los colores RGB, para ello hemos usado las funciones `cv2.cvtColor` y `np.zeros_like` y le hemos asignado a la imagen gris los diferentes canales rojo, verde y azul.

Para el modo 2 se decidió usar una técnica de la práctica 2, la detección de bordes por sobel y canny.  
Primero usaremos sobel con las funciones `sobel_frame` y `cv2.cvtColor` que harán que una vez que tengamos la imagen de sobel, luego dicha imagen binaria pase a tener 3 canales, necesario para luego usar la función `cv2.putText`.  
Al volver a apretar el modo 2 usaremos canny con sus funciones `canny_frame` y `cv2.cvtColor` para obtener lo mismo que con el procedimiento de sobel, el frame gris con detectores de borde gracias a canny y luego se pasa a 3 canales para poder mostrarla con texto.

Finalmente en el último modo aplicaremos una substracción de fondo.  
Usamos la función `eliminadorFondo.apply` con la que obtendremos una imagen en escala de grises donde un píxel con valor 0 se considerará fondo, uno con valor 255 será foreground(objetos en movimiento) y uno con valor 127 será sombra.  
Usamos también la función `cv2.cvtColor` para poder colorear y mostrar texto. Luego creamos una máscara booleana donde los píxeles con valor 127 se detectan como true y se pintan de un color tenue para que sean visibles en la imagen.  
Por último con la función `cv2.bitwise_and` que aplica la máscara sobre la imagen original, dejando visible solo los objetos en movimiento.

También está incluido el manejo de las teclas para los modos, con la función `cv2.waitKey()` capturamos la tecla pulsada.  
Si la tecla pulsada es 2, confirma en qué modo está, si ya está en el modo 2, se cambia la vista interna, por el contrario, se entraría en el modo 2.  
Si la tecla pulsada es la 1, se cicla entre los modos rojo, verde y azul, si la tecla pulsada no es la 1, se entraría en el modo 1.  
Por último si se pulsa la tecla 0 o 3 entraríamos en dichos modos.

### Ejercicio 4
Objetivo -> proponer un demostrador reinterpretando la parte de procesamiento de la imagen, tomando como punto de partida alguna de dichas instalaciones.  
Yo tomaré como referencia el "Messa di voce".

Para elaborar este demostrador primero declaramos algunas variables; la imagen donde se dibujarán las partículas, la lista donde se guardan las partículas activas, la paleta de colores para las partículas y el frame anterior para detectar el movimiento.

Luego pasamos a capturar las imágenes girando el frame horizontalmente para hacer más intuitiva la interacción. También inicializamos la imagen donde se dibujarán las partículas.

Seguimos convirtiendo la imagen a escala de grises y aplicando el desenfoque gaussiano (`cv2.cvtColor`, `cv2.GaussianBlur`) para facilitar la detección de movimiento y reducir el ruido.  
Una vez transformada la imagen, calculamos la diferencia entre el frame actual y el anterior `cv2.absdiff`, pues en donde haya movimiento, los píxeles tendrán valor alto.  
También convertimos las diferencias grandes en blanco y el resto en negro con `cv2.threshold` y expandimos las regiones blancas, rellenando huecos pequeños con `cv2.dilate`.

Las partículas se generarán de la siguiente manera:  
- Primero con `cv2.findContours` se generan contornos que representan objetos o regiones en movimiento/regiones blancas.  
- Calculamos las áreas de estas regiones con `cv2.contourArea` y si son muy pequeñas se ignoran.  
- Luego con `cv2.boundingRect` obtenemos el rectángulo que encierra el contorno.  
- Luego calculamos los centros de los contornos y los usaremos para generar radios proporcionales al tamaño de contorno.  
- Para acabar la creación de las partículas, las coloreamos de manera aleatoria con `np.random.randint` y las guardamos como una lista de 5 elementos `[cx, cy, radio, vida, color]`.

Para hacer el efecto de que las partículas se desvanecen aplicamos decay sobre la imagen overlay, el radio de los círculos crece con el tiempo hasta que sus vidas llegan a 0 y se eliminan.  
Para dibujar los círculos y sus bordes que representan las partículas usamos `cv2.circle`.

Finalmente mezclamos la imagen original con el overlay de partículas con `cv2.addWeighted`, con un efecto semi transparente gracias a las ponderaciones de 0.6 del frame y 0.8 del overlay marcados en la misma función.

<!-- - [Práctica 3](P3/README.md) -->
<!-- - [Práctica 4](P4/README.md) -->
<!-- - [Práctica 5](P5/README.md) -->
<!-- - [Práctica 6](P6/README.md) -->
<!-- - [Práctica 7](P7/README.md) -->
<!-- - [Trabajo](Trabajo/README.md) -->

***
Obra bajo licencia de Creative Commons Reconocimiento - No Comercial 4.0 Internacional
