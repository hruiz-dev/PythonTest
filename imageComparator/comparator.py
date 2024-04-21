import cv2
import os
import shutil

def comparar_imagenes(img1_path, img2_path):
    img_path = "./imagenes/"
    # Cargar las dos imágenes
    img1 = cv2.imread(img_path + img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path + img2_path, cv2.IMREAD_GRAYSCALE)

    # Inicializar el detector ORB
    orb = cv2.ORB_create()

    # Detectar y calcular las características
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Inicializar el matcher de fuerza bruta
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Realizar el matching
    matches = bf.match(des1, des2)

    # Ordenar los matches por distancia
    matches = sorted(matches, key = lambda x:x.distance)

    # Devolver el número de matches
    return len(matches)

# Ejemplo de uso
img1 = "coche2.jpg"
img2 = "coche1.jpg"
matches = comparar_imagenes(img1, img2)
print("Número de matches entre las imágenes:", matches)
# if matches > 100:
#     os.makedirs("./coincidencias/1", exist_ok=True)
#     shutil.move("./imagenes/" + img1, "./coincidencias/1/" + img1)
#     shutil.move("./imagenes/" + img2, "./coincidencias/1/" + img2)