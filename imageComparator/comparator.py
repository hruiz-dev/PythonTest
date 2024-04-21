import cv2
import numpy as np

def comparar_imagenes(img1_path, img2_path):
    # Cargar las imágenes en escala de grises
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Calcular los histogramas de las imágenes
    hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])

    # Normalizar los histogramas
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Calcular la correlación entre los histogramas
    correlacion = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # Devolver la correlación
    return correlacion

# Ejemplo de uso
img1_path = "./imagenes/coche1.jpg"
img2_path = "./imagenes/coche3.jpg"
correlacion = comparar_imagenes(img1_path, img2_path)
print("Correlación entre las imágenes:", correlacion)