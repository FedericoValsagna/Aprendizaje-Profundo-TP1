import os
import numpy as np
from PIL import Image

FOLDERNAME = "Imagenes"
TARGET_SIZE = (50, 50)  # (Ancho, Alto) estándar para todas las imágenes


def transformación(matrix):
    # Flatten del arreglo
    matrix = matrix.flatten()

    # Si los valores vienen en rango 0-255 o booleanos,
    # mapeamos los píxeles claros (> 128) a 1 y los oscuros a -1.
    return np.where(matrix > 128, 1, -1).astype(np.int8)


def transform_images():
    Xs = []
    for file in imagenes():
        matriz = bmp_to_numpy_array(os.path.join(FOLDERNAME, file))
        Xs.append(transformación(matriz))
    return Xs


def bmp_to_numpy_array(file_path):
    imagen = Image.open(file_path).convert('L')  # Convertimos a escala de grises
    imagen = imagen.resize(TARGET_SIZE)          # Redimensionamos a 50x50

    matriz = np.array(imagen)

    # Si la imagen es booleana, convertimos True->255 y False->0
    if matriz.dtype == bool:
        matriz = matriz.astype(np.uint8) * 255
    else:
        matriz = matriz.astype(np.uint8)

    return matriz


def numpy_array_to_bmp(matriz, file_path, target_shape=TARGET_SIZE):
    """
    Convierte un vector bipolar 1D (con valores -1 y 1) o una matriz 2D
    de vuelta a formato de imagen y la guarda como BMP.
    """
    # Reestructurar a 2D si viene como un vector aplanado (1D)
    if matriz.ndim == 1:
        matriz = matriz.reshape((target_shape[1], target_shape[0]))

    # Convertir valores bipolares (-1, 1) a escala de grises uint8 (0, 255)
    img_uint8 = np.where(matriz > 0, 255, 0).astype(np.uint8)

    imagen = Image.fromarray(img_uint8)
    imagen.save(file_path)


def imagenes():
    return [file for file in os.listdir(FOLDERNAME) if file.lower().endswith(".bmp")]

def image_name_from_id(id):
    images = ["Perro", "Quijote""Torero", "Paloma", "Torero","Panda", "Anonymous"]
    return images[id]