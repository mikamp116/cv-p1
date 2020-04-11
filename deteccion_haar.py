import cv2
from matplotlib import pyplot as plt
import deteccion_orb as orbdet

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html


def detect(gray_images, color_images, coches_cascade, matriculas_cascade, scale_factor, min_neighbors):
    """Detecta el frontal y la matricula de los coches y devuelve una lista de imágenes con las zonas detectadas"""
    detected_images = []

    for i in range(len(gray_images)):
        gray = gray_images[i]
        color = color_images[i]

        coche = coches_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        for (x, y, w, h) in coche:
            color = cv2.rectangle(color, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = color[y:y + h, x:x + w]
            matricula = matriculas_cascade.detectMultiScale(roi_gray, scale_factor, min_neighbors)
            for (ex, ey, ew, eh) in matricula:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 4)

        matricula = matriculas_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        for (x, y, w, h) in matricula:
            color = cv2.rectangle(color, (x, y), (x + w, y + h), (0, 0, 255), 2)

        detected_images.append(color)

    return detected_images


def show(images):
    for i in range(len(images)):
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
        plt.title("Imagen " + str(i + 1))
        plt.show()


def main(scale_factor, min_neighbors):
    coches_cascade = cv2.CascadeClassifier('haar_opencv_4.1-4.2/coches.xml')
    matriculas_cascade = cv2.CascadeClassifier('haar_opencv_4.1-4.2/matriculas.xml')

    test_images_gray = orbdet.load('test')
    test_images_color = orbdet.load_color('test')

    frontales = detect(test_images_gray, test_images_color, coches_cascade, matriculas_cascade, scale_factor,
                       min_neighbors)
    show(frontales)


if __name__ == "__main__":
    SCALE_FACTOR = 1.3
    MIN_NEIGHBORS = 5

    main(SCALE_FACTOR, MIN_NEIGHBORS)
