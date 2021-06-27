# Computer Vision - Practica 1: Detección de objetos

## 0. Demo

![cv-p1](https://user-images.githubusercontent.com/48054735/123549640-26a84800-d76a-11eb-83d0-1becc81ee748.gif)

## 1. Detección de coches mediante puntos de interés

Este apartado consiste en la construcción de un sistema que permita la detección de la posición de un coche dentro de una
imagen. Las imágenes de coches se encontrarán en forma frontal. Se suministra un directorio <code>/training</code> 
con 48 imágenes de coches en los que se ha recortado la parte más importante, la cual está libre de reflejos para que 
se use como muestra de aprendizaje.
Además, se suministra otro directorio <code>/test</code> con imágenes de 33 coches para que se usen como
muestra de prueba. Sobre este segundo grupo de imágenes se obtendrán las estadísticas de funcionamiento del sistema.

Para construir el sistema se utilizará un enfoque basado en la detección de puntos de interés 
(Harris, FAST, SIFT, etc.), seguido de una descripción de los mismos (BRIEF, SIFT, etc.) y votación a la Hough.

## 2. Detección de coches usando cv2.CascadeClassifier

Este apartado consiste en la construcción de un sistema basado en el detector de P.Viola y M. Jones 
(<code>cv2.CascadeClassifier</code> en OpenCV) que permita la detección de la posición de un coche dentro de una imagen. 
Las imágenes de coches se encontrarán en forma frontal, y se utilizarán las imágenes de coches del directorio 
<code>/test</code>. Para ello, se se suministra un fichero <code>coches.xml</code> que corresponde con un
clasificador de coche/no-coche ya entrenado.

## 3. Detección del coche en secuencias de vídeo

Este apartado consiste en desarrollar un sistema que utilice los detectores desarrollados en los apartados anteriores 
sobre las secuencias de vídeo suministradas. 

## 4. Ejecución

La práctica deberá ejecutarse sobre <code>Python 3.7.X</code> y <code>OpenCV 4.2</code> y consistirá en 3 ficheros python:
- <code>deteccion_orb.py</code>
- <code>detección_haar.py</code>
- <code>deteccion_video.py</code>

Para ejecutar la práctica deberá escribirse en la consola de comandos *python* seguido del nombre del
fichero sin ningún otro parámetro adicional. **Se supondrá que los directorios *test*, *train* y *haar* están en
el mismo directorio que los ficheros python.** Al ejecutar estos ficheros python se mostrará por pantalla
el resultado sobre cada una de las imágenes o vídeos de test.

## 5. Implementación, descripción del código y resultados obtenidos

Véase el documento *memoria.pdf*
