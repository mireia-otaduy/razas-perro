import cv2 as cv
import numpy as np
import sys
import pickle

import preparar_dataset as P
import descriptores as D
import clasificar as CL

# ------------------------------------------------------------
# reconocer la raza de un perro en una foto nueva (version 2: 120 razas)
#
# pasos (los mismos que se hicieron con el dataset):
#   1. recortar al perro: saliency + grabcut, fondo negro, 256x256   (preparar_dataset.py)
#   2. sacar los descriptores: color, hog, lbp, bow, fisher y color local (descriptores.py)
#   3. el clasificador (lda) da una nota por raza                    (clasificar.py)
#   4. respuesta: las 3 razas que mas se parecen, de mas a menos
#
# uso:  python reconocer_raza.py C:/ruta/foto_de_mi_perro.jpg
#
# para que funcione bien la foto tiene que ser asi: un solo perro, cuerpo entero o casi,
# de frente, con la cara bien visible y sin mucha gente ni objetos alrededor
#
# ojo: solo conoce las 120 razas del dataset. con otra raza (o un mestizo) dira las 3 que mas se parezcan
# con 120 razas el acierto es bajo (en test: la raza buena sale la primera en 37% de las fotos
# y entre las 3 primeras en 55%), asi que hay que mirar siempre las 3 opciones
# ------------------------------------------------------------

foto = "datos/dataset_stanford/images/n02099601-golden_retriever/n02099601_100.jpg"   # foto por defecto
modelo = "modelo/modelo.pkl"        # modelo entrenado por clasificar.py
MOSTRAR = "--sin-ventanas" not in sys.argv                # ensenar la foto y el recorte en ventanas


# cargar el modelo entrenado (pesos, escalado, razas, temperatura y umbral)
def loadModel(path:str) -> dict:
    f = open(path, "rb")
    model = pickle.load(f)
    f.close()
    return model


# nombre bonito de una raza
def prettyName(breed:str) -> str:
    return breed.replace("_", " ")


# probabilidad de cada raza para una foto
# devuelve tambien el recorte y si el recorte es dudoso
def predictBreed(img:cv.Mat, model:dict) -> tuple:
    # paso 1: recorte del perro (fondo negro, 256x256) y su mascara
    result, result_mask, doubt, box = P.processImage(img)
    # paso 2: descriptores (una sola fila porque es una sola foto)
    d = D.describeImage(result, result_mask)
    data = {}
    for key in d:
        data[key] = np.array([d[key]], np.float32)
    # paso 3: nota de cada raza y paso a probabilidad
    X = CL.buildMatrix(data, np.array([0]), model["keys"], model["scaler"])
    S = CL.scoreLDA(X, model["W"], model["bias"], model["mu"])
    proba = CL.getProba(S, model["t"])[0]
    return proba, result, doubt


def main():
    path = foto
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) > 0:
        path = args[0]
    img = cv.imread(path)
    if img is None:
        sys.exit("Could not read the image.")
    model = loadModel(modelo)

    proba, result, doubt = predictBreed(img, model)
    print("foto:", path)
    # las 3 razas mas probables
    print("las 3 razas que mas se parecen (de mas a menos):")
    for i in np.argsort(-proba)[:3]:
        print("  ", prettyName(model["breeds"][i]))
    if doubt:
        print("aviso: el recorte del perro no es muy fiable, prueba con otra foto (un solo perro, de frente y sin mucho fondo)")

    if MOSTRAR:
        cv.imshow("foto", P.resizeMax(img, 512))
        cv.imshow("recorte", result)
        cv.waitKey(0)


if __name__ == "__main__":
    main()
