import cv2 as cv
import numpy as np
import os
import sys
import time
import shutil
from multiprocessing import Pool

# ------------------------------------------------------------
# preparar el dataset de perros (version 2: stanford dogs)
#
# que hace este programa con cada foto de perro:
#   1. saliency: busca que parte de la foto llama mas la atencion (el perro)
#   2. rectangulo: saca un rectangulo alrededor de esa zona
#   3. grabcut: separa el perro del fondo dentro del rectangulo (como en la practica 4.2)
#   4. fondo negro: todo lo que no es perro se pone en negro
#   5. recorte: nos quedamos solo con el trozo del perro
#   6. resize: lo dejamos en 256x256 sin estirarlo (si sobra sitio, negro)
#   7. mascara: guardamos tambien una foto en blanco y negro que dice donde esta el perro
#
# es el mismo recorte que en la version 1. lo unico nuevo es que ademas apuntamos
# en recortes.csv la caja del recorte (para medir luego los desastres)
#
# las 3 razas que vienen de oxford (american bulldog, havanese y shiba inu) ya estan recortadas
# con este mismo programa en la version 1: solo se copian
#
# se puede parar y volver a ejecutar: salta las fotos que ya estan hechas
# uso: python preparar_dataset.py [segundos]   (los segundos son un limite de tiempo, opcional)
# ------------------------------------------------------------

carpeta = "datos/dataset_stanford/images"    # fotos originales
lista = "datos/lista_fotos.csv"              # la hace preparar_lista.py
salida = "datos/perros_256"                  # aqui se guardan las nuevas
salida_mask = "datos/perros_256_mask"        # y aqui la mascara de cada una (blanco = perro)
recortes = "datos/recortes.csv"              # caja del recorte de cada foto
carpeta_v1 = "C:/vision/proyecto_v1/datos"                         # recortes de la version 1
SIZE = 256    # tamano final de las fotos (256x256)
WORK = 256    # saliency y grabcut se hacen con la foto a este tamano para ir mas rapido


# poner la foto con el lado mas largo = size (tema 3, resize)
def resizeMax(img:cv.Mat, size:int) -> cv.Mat:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if scale >= 1:
        return img    # si ya es pequena la dejamos igual
    return cv.resize(img, (int(round(w*scale)), int(round(h*scale))), interpolation=cv.INTER_AREA)


# paso 1: mapa de saliency (zonas claras = llaman mas la atencion, de 0 a 255)
def getSaliencyMap(img:cv.Mat) -> cv.Mat:
    saliency = cv.saliency.StaticSaliencyFineGrained_create()
    ret, sal_map = saliency.computeSaliency(img)
    sal_map = cv.normalize(sal_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    # blur gaussiano para quitar puntitos sueltos (tema 4)
    sal_map = cv.GaussianBlur(sal_map, (0,0), 5)
    # el perro suele estar hacia el centro: damos mas peso al centro de la foto
    rows, cols = sal_map.shape
    gx = cv.getGaussianKernel(cols, cols/2)
    gy = cv.getGaussianKernel(rows, rows/2)
    center = gy * gx.T
    center = center / center.max()
    sal_map = sal_map * (0.5 + 0.5*center)
    sal_map = cv.normalize(sal_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return sal_map


# paso 2: rectangulo del perro (practica 4.1: umbral + findContours + boundingRect)
def getRect(sal_map:cv.Mat, thr:float, margin:float) -> tuple:
    rows, cols = sal_map.shape
    # umbral: nos quedamos con lo que es mas claro que thr
    ret, thresh = cv.threshold(sal_map, thr, 255, cv.THRESH_BINARY)
    # close: une los trozos que estan cerca
    kernel = np.ones((3,3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # si no encuentra nada: rectangulo grande en el centro
        return (int(cols*0.1), int(rows*0.1), int(cols*0.8), int(rows*0.8))
    # nos quedamos con los contornos grandes
    areas = [cv.contourArea(cnt) for cnt in contours]
    big = [cnt for cnt, a in zip(contours, areas) if a > 0.15*max(areas)]
    x, y, w, h = cv.boundingRect(np.vstack(big))
    # un poco de margen para no cortar al perro
    mx, my = int(w*margin), int(h*margin)
    x0, y0 = max(x-mx, 0), max(y-my, 0)
    x1, y1 = min(x+w+mx, cols), min(y+h+my, rows)
    return (x0, y0, x1-x0, y1-y0)


# limpiar la mascara del perro: quita puntitos, tapa agujeros y deja solo el trozo mas grande
def cleanMask(mask:cv.Mat) -> cv.Mat:
    kernel = np.ones((3,3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)           # open: quita puntitos
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)    # close: tapa huecos pequenos
    n, labels, stats, centroids = cv.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    # nos quedamos con el trozo mas grande
    biggest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    mask = (labels == biggest).astype(np.uint8)
    # rellenamos los agujeros con el contorno exterior
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv.drawContours(filled, contours, -1, 1, -1)
    return filled


# paso 3: grabcut (practica 4.2, igual que en clase: solo se le da el rectangulo)
def applyGrabCut(img:cv.Mat, rect:tuple) -> cv.Mat:
    # borde extra a la foto para que siempre haya fondo fuera del rectangulo
    pad = 10
    img_pad = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    rect_pad = (rect[0]+pad, rect[1]+pad, rect[2], rect[3])
    mask = np.zeros(img_pad.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    # grabcut con 5 iteraciones
    cv.grabCut(img_pad, mask, rect_pad, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    # 0 y 2 = fondo, 1 y 3 = perro (igual que en la practica 4.2)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    return cleanMask(mask2[pad:-pad, pad:-pad])     # quitamos el borde extra


# si el perro toca el borde del rectangulo es que se ha quedado corto: lo hacemos mas grande
def expandRect(rect:tuple, mask:cv.Mat) -> tuple:
    rows, cols = mask.shape
    x, y, w, h = rect
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return rect
    x0, y0, x1, y1 = x, y, x+w, y+h
    ex, ey = int(0.2*w), int(0.2*h)    # lo agrandamos un 20%
    if xs.min() <= x+3 and x > 3:           x0 = max(x-ex, 0)          # toca por la izquierda
    if xs.max() >= x+w-4 and x+w < cols-3:  x1 = min(x+w+ex, cols)     # toca por la derecha
    if ys.min() <= y+3 and y > 3:           y0 = max(y-ey, 0)          # toca por arriba
    if ys.max() >= y+h-4 and y+h < rows-3:  y1 = min(y+h+ey, rows)     # toca por abajo
    return (x0, y0, x1-x0, y1-y0)


# pasos 4, 5 y 6: fondo negro, recorte y resize a size x size sin estirar
def cropAndResize(img:cv.Mat, mask:cv.Mat, size:int=SIZE) -> cv.Mat:
    dog = img * mask[:, :, np.newaxis]     # todo lo que no es perro = negro
    # recorte (roi): la caja mas pequena que tiene al perro dentro
    ys, xs = np.where(mask > 0)
    dog = dog[ys.min():ys.max()+1, xs.min():xs.max()+1]
    # el lado mas largo pasa a medir size (asi no se estira)
    h, w = dog.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(int(round(w*scale)), 1), max(int(round(h*scale)), 1)
    dog = cv.resize(dog, (new_w, new_h), interpolation=cv.INTER_AREA)
    # lienzo negro size x size y ponemos al perro en el centro
    result = np.zeros((size, size, 3), np.uint8)
    top, left = (size-new_h)//2, (size-new_w)//2
    result[top:top+new_h, left:left+new_w] = dog
    return result


# todo junto para una foto
# devuelve: la foto nueva, su mascara, si dudamos del resultado (True/False) y la caja del recorte
def processImage(img:cv.Mat) -> tuple:
    # semilla fija: grabcut usa numeros al azar por dentro, asi sale siempre el mismo recorte
    cv.setRNGSeed(0)
    img0 = img                     # foto original (grande)
    img = resizeMax(img0, WORK)    # foto pequena para buscar al perro
    rows, cols = img.shape[:2]
    sal_map = getSaliencyMap(img)
    # otsu (practica 4.3) nos da el umbral que separa lo muy claro de lo oscuro
    otsu, thresh = cv.threshold(sal_map, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # rectangulo con el saliency (con bastante margen para no cortar al perro)
    rect = getRect(sal_map, 0.4*otsu, 0.15)
    mask = applyGrabCut(img, rect)
    # si el perro toca el borde del rectangulo, lo agrandamos y repetimos
    new_rect = expandRect(rect, mask)
    if new_rect != rect:
        rect = new_rect
        mask = applyGrabCut(img, rect)
    # ratio = que parte del rectangulo es perro (si es muy poco o casi todo, algo fallo)
    ratio = mask.sum() / (rect[2]*rect[3])

    # si ha salido mal, probamos con un rectangulo casi tan grande como la foto
    if ratio < 0.2 or ratio > 0.9:
        rect = (int(cols*0.05), int(rows*0.05), int(cols*0.9), int(rows*0.9))
        mask = applyGrabCut(img, rect)
        ratio = mask.sum() / (rect[2]*rect[3])

    # si sigue mal, lo apuntamos como dudosa
    doubt = ratio < 0.2 or ratio > 0.9
    # y si grabcut no ha encontrado casi nada, usamos el rectangulo entero
    if mask.sum() < 0.05*rect[2]*rect[3]:
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 1
    # la mascara se ha calculado con la foto pequena: la ponemos al tamano de la foto original
    # y recortamos al perro de la original, asi no se pierde detalle (ojos, hocico, pelo)
    mask = cv.resize(mask, (img0.shape[1], img0.shape[0]), interpolation=cv.INTER_NEAREST)
    # caja del recorte en la foto original (x0, y0, x1, y1)
    ys, xs = np.where(mask > 0)
    box = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
    result = cropAndResize(img0, mask)
    # hacemos lo mismo con la mascara: asi sabemos que pixeles de la foto final son perro
    # (sirve luego para calcular los descriptores solo dentro del perro)
    mask_img = cv.merge([mask*255, mask*255, mask*255])
    result_mask = cropAndResize(mask_img, mask)[:, :, 0]
    ret, result_mask = cv.threshold(result_mask, 127, 255, cv.THRESH_BINARY)
    return result, result_mask, doubt, box


# procesar una foto del disco y guardarla (la usa el Pool)
# task = (carpeta de la raza, nombre de la foto sin .jpg)
def processFile(task:tuple) -> tuple:
    folder, name = task
    img = cv.imread(carpeta + "/" + folder + "/" + name + ".jpg")
    if img is None:
        return name, True, (0, 0, 0, 0)    # foto rota
    result, result_mask, doubt, box = processImage(img)
    # guardamos con nombre temporal y luego lo renombramos (asi no quedan fotos a medias)
    final_mask = salida_mask + "/" + name + ".png"
    cv.imwrite(final_mask + ".tmp.png", result_mask)
    os.replace(final_mask + ".tmp.png", final_mask)
    final = salida + "/" + name + ".png"
    cv.imwrite(final + ".tmp.png", result)
    os.replace(final + ".tmp.png", final)
    return name, doubt, box


# leer la lista de fotos (la hace preparar_lista.py): una fila por foto
def readList() -> list:
    rows = []
    for line in open(lista).read().split("\n")[1:]:
        if line:
            name, breed, split, source, folder, width, height, boxes = line.split(",")
            rows.append((name, breed, source, folder))
    return rows


# las razas de oxford ya estan recortadas en la version 1: copiamos foto y mascara
def copyOxford(rows:list) -> None:
    n = 0
    for name, breed, source, folder in rows:
        if source == "oxford" and not os.path.exists(salida + "/" + name + ".png"):
            shutil.copy(carpeta_v1 + "/perros_256_mask/" + name + ".png", salida_mask + "/" + name + ".png")
            shutil.copy(carpeta_v1 + "/perros_256/" + name + ".png", salida + "/" + name + ".png")
            n += 1
    print("fotos copiadas de oxford:", n, flush=True)


def main():
    cv.setNumThreads(1)
    t_end = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(salida, exist_ok=True)
    os.makedirs(salida_mask, exist_ok=True)
    rows = readList()
    copyOxford(rows)

    # fotos de stanford que faltan (hecha = tiene su foto y su linea en recortes.csv)
    done = set()
    if os.path.exists(recortes):
        done = set(line.split(",")[0] for line in open(recortes).read().split("\n") if line)
    tasks = [(folder, name) for name, breed, source, folder in rows if source == "stanford" and name not in done]
    print("fotos de stanford por hacer:", len(tasks), "| ya hechas:", len(done), flush=True)

    # recortes.csv: name,x0,y0,x1,y1,dudosa  (dudosa = el programa no esta seguro del recorte)
    f = open(recortes, "a")
    n = 0
    # el Pool usa varios nucleos a la vez (solo para ir mas rapido)
    # trabajamos por lotes pequenos para poder parar cuando se acaba el tiempo
    batch = 8 * os.cpu_count()
    with Pool(os.cpu_count()) as pool:
        for i in range(0, len(tasks), batch):
            if t_end is not None and time.time() > t_end:
                break
            for name, doubt, box in pool.imap_unordered(processFile, tasks[i:i+batch]):
                f.write(name + "," + ",".join(str(v) for v in box) + "," + str(int(doubt)) + "\n")
                f.flush()
                n += 1
            if (i // batch) % 25 == 0:
                print(n, "hechas", flush=True)
    f.close()
    print("hechas en esta tanda:", n, "| faltan:", len(tasks) - n, flush=True)


if __name__ == "__main__":
    main()
