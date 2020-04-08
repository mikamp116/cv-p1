import cv2
from matplotlib import pyplot as plt
import deteccion_orb as orbdet

coches_cascade = cv2.CascadeClassifier('haar_opencv_4.1-4.2/coches.xml')
matricula_cascade = cv2.CascadeClassifier('haar_opencv_4.1-4.2/matriculas.xml')

test_images = orbdet.load('test')

for image in test_images:
    coche = coches_cascade.detectMultiScale(image, 1.3, 5)
    matricula = matricula_cascade.detectMultiScale(image, 1.3, 5)

    for (x, y, w, h) in coche:
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in matricula:
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    plt.imshow(image)
    plt.show()
