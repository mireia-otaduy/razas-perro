import cv2 as cv
import numpy as np
import os
import sys
import time
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# ------------------------------------------------------------
# descriptores (version 2): convertir cada foto de perro en una lista de numeros
#
# de cada foto 256x256 (ya recortada, con fondo negro) sacamos:
#   color  -> histograma de color hsv                      (practicas 1.3 y 7.4)
#   hog    -> histogramas de direcciones de los bordes     (tema 6)
#   lbp    -> textura del pelo (local binary patterns)     (tema 7, solo nombrado)
#   lbp2   -> lo mismo, pero mirando vecinos un poco mas lejos
#   bow    -> bag of words con sift + k-means              (practica 5.1, temas 5 y 7)
#   fisher -> fisher vectors con sift (EXTRA, no visto en clase)
#   lcolor -> fisher vectors del color local: parches de 16x16 y su color en lab (EXTRA, no visto en clase)
#
# fisher vectors: como el bag of words, pero en vez de solo contar palabras guarda
# tambien cuanto se aleja cada descriptor del centro de su grupo. ocupa mas, pero
# dice mucho mas de cada foto. (usa pca y una mezcla de gaussianas en vez de k-means)
#
# todos se calculan solo dentro de la mascara del perro, para no describir el fondo negro
# de cada foto se guarda un fichero .npz con todos sus descriptores
# (las fotos de entrenamiento guardan ademas los descriptores de la foto en espejo)
#
# se puede parar y volver a ejecutar: salta lo que ya esta hecho
# uso: python descriptores.py [segundos]   (los segundos son un limite de tiempo, opcional)
# ------------------------------------------------------------

lista = "datos/lista_fotos.csv"
carpeta = "datos/perros_256"              # fotos recortadas
carpeta_mask = "datos/perros_256_mask"    # sus mascaras (blanco = perro)
salida = "datos/descriptores"             # aqui va un .npz por foto
descartadas = "datos/descartadas"                     # fotos quitadas del dataset
descartadas_mask = "datos/descartadas_mask"
salida_descartadas = "datos/descriptores_descartadas"
muestras = "modelo/muestras_sift.npy"     # descriptores sift para crear los vocabularios
vocabulario = "modelo/vocabulario.npy"    # palabras visuales del bag of words
fisher_modelo = "modelo/fisher.npz"       # pca y gaussianas de los fisher vectors
fisher_color_modelo = "modelo/fisher_color.npz"   # gaussianas del color local
SIZE = 256    # tamano de las fotos
K = 1000      # numero de palabras del vocabulario
D_PCA = 64    # numeros por descriptor sift despues del pca (fisher)
K_GMM = 32    # numero de gaussianas (fisher)
PARCHE = 16   # tamano de los parches de color local
PASO = 8      # cada cuantos pixeles ponemos un parche
vocab = None  # el vocabulario se carga la primera vez que hace falta
fisher = None # lo mismo con los datos de fisher
fisher_color = None   # lo mismo con los datos del color local


# lista de fotos: diccionario nombre -> (raza, train o test)
def readList() -> dict:
    info = {}
    for line in open(lista).read().split("\n")[1:]:
        if line:
            name, breed, split = line.split(",")[:3]
            info[name] = (breed, split)
    return info


# fotos de una carpeta, lista de razas (las 120) y raza (como numero) y grupo (train o test) de cada foto
def listDataset(folder:str) -> tuple:
    info = readList()
    names = [n[:-4] for n in sorted(os.listdir(folder)) if n.endswith(".png") and n[:-4] in info]
    breeds = sorted(set(breed for breed, split in info.values()))
    labels = np.array([breeds.index(info[n][0]) for n in names])
    splits = np.array([info[n][1] for n in names])
    return names, breeds, labels, splits


# color: histograma hsv solo con los pixeles del perro (calcHist con mascara, practica 7.4)
def getColorHist(img:cv.Mat, mask:cv.Mat) -> np.ndarray:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # histograma de 2 dimensiones: tono (18 trozos) y saturacion (8 trozos)
    hist = cv.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256]).flatten()
    # histograma del brillo (8 trozos)
    histV = cv.calcHist([hsv], [2], mask, [8], [0, 256]).flatten()
    hist = np.concatenate([hist, histV])
    # dividimos entre el numero de pixeles del perro (asi no importa el tamano)
    return hist / max(cv.countNonZero(mask), 1)


# hog: histogramas de la direccion de los bordes por zonas (tema 6)
def getHOG(img:cv.Mat) -> np.ndarray:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ventana 256x256, bloques de 128, paso de 64, celdas de 64 y 9 direcciones
    hog = cv.HOGDescriptor((SIZE, SIZE), (128, 128), (64, 64), (64, 64), 9)
    return hog.compute(gray).flatten()


# lbp de un canal: cada pixel se compara con sus 8 vecinos (los que estan a r pixeles)
# si el vecino es mayor o igual se apunta un 1, si no un 0 -> sale un numero de 0 a 255
def getLBPImage(gray:cv.Mat, r:int) -> cv.Mat:
    gray = gray.astype(np.int16)
    n = SIZE - 2*r                      # la imagen lbp pierde r pixeles por cada borde
    center = gray[r:r+n, r:r+n]
    lbp = np.zeros(center.shape, np.uint8)
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(neighbours):
        neighbour = gray[r+dy*r:r+dy*r+n, r+dx*r:r+dx*r+n]
        lbp = lbp | ((neighbour >= center).astype(np.uint8) << bit)
    return lbp


# lbp: histograma de esos numeros en los 3 canales hsv (describe la textura del pelo)
# r = a cuantos pixeles estan los vecinos (1 = textura fina, 2 = textura un poco mas gruesa)
def getLBP(img:cv.Mat, mask:cv.Mat, r:int) -> np.ndarray:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask_in = np.ascontiguousarray(mask[r:SIZE-r, r:SIZE-r])
    hists = []
    for c in range(3):
        lbp = getLBPImage(hsv[:, :, c], r)
        hist = cv.calcHist([lbp], [0], mask_in, [256], [0, 256]).flatten()
        hists.append(hist / max(hist.sum(), 1))
    return np.concatenate(hists)


# sift denso: en vez de dejar que sift busque los keypoints, los ponemos nosotros
# en una rejilla (cada 8 pixeles, 3 tamanos) y solo dentro del perro
# asi cada foto tiene muchos descriptores sift (128 numeros cada uno, practica 5.1)
def getDenseSIFT(img:cv.Mat, mask:cv.Mat) -> np.ndarray:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp = []
    for y in range(4, SIZE, 8):
        for x in range(4, SIZE, 8):
            if mask[y, x] > 0:
                for size in (8, 16, 28):
                    kp.append(cv.KeyPoint(float(x), float(y), float(size)))
    sift = cv.SIFT_create()
    kp, des = sift.compute(gray, kp)
    if des is None:
        return np.zeros((0, 128), np.float32)
    return des


# bag of words (tema 7): a cada descriptor sift le toca la palabra del vocabulario
# que mas se le parece, y contamos cuantas veces sale cada palabra (histograma)
def getBoW(des:np.ndarray, vocab:np.ndarray) -> np.ndarray:
    hist = np.zeros(len(vocab), np.float32)
    if len(des) == 0:
        return hist
    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.match(des, vocab)
    for m in matches:
        hist[m.trainIdx] += 1
    return hist / hist.sum()



# fisher vectors (extra, no visto en clase)
#   1. rootsift: sift dividido entre su suma y con raiz cuadrada (funciona mejor que el sift normal)
#   2. pca: de 128 numeros por descriptor a 64
#   3. cada descriptor "reparte" su peso entre las K_GMM gaussianas (la que mas se le parece se lleva mas)
#   4. por cada gaussiana guardamos cuanto se aleja la media de los descriptores de su centro
#      y cuanto cambia su varianza -> 2 * K_GMM * D_PCA numeros
#   5. raiz cuadrada con signo y normalizar (asi no mandan los valores muy grandes)
def getFisher(des:np.ndarray, fisher:dict) -> np.ndarray:
    size = 2 * K_GMM * D_PCA
    if len(des) < 2:
        return np.zeros(size, np.float32)
    x = np.sqrt(des / np.maximum(des.sum(axis=1, keepdims=True), 1e-7))
    x = (x - fisher["mean_pca"]) @ fisher["components"].T
    w, mu, var = fisher["weights"], fisher["means"], fisher["variances"]
    # probabilidad de que cada descriptor sea de cada gaussiana (paso 3)
    maha = (x*x) @ (1/var).T - 2 * x @ (mu/var).T + (mu*mu/var).sum(axis=1)[np.newaxis, :]
    logp = -0.5 * (maha + np.log(var).sum(axis=1)[np.newaxis, :]) + np.log(w)[np.newaxis, :]
    logp = logp - logp.max(axis=1, keepdims=True)
    gamma = np.exp(logp)
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    # paso 4
    n = len(x)
    s0 = gamma.sum(axis=0)
    s1 = gamma.T @ x
    s2 = gamma.T @ (x*x)
    g_mu = (s1 - s0[:, np.newaxis]*mu) / (np.sqrt(var) * np.sqrt(w)[:, np.newaxis] * n)
    g_var = (s2 - 2*mu*s1 + (mu*mu - var)*s0[:, np.newaxis]) / (var * np.sqrt(2*w)[:, np.newaxis] * n)
    fv = np.concatenate([g_mu.flatten(), g_var.flatten()])
    # paso 5
    fv = np.sign(fv) * np.sqrt(np.abs(fv))
    return (fv / max(np.linalg.norm(fv), 1e-7)).astype(np.float32)


# color local: en vez de un solo histograma de toda la foto, miramos el color de muchos parches pequenos
# parches de 16x16 cada 8 pixeles; solo los que estan casi todos dentro del perro
# de cada parche sacamos 12 numeros (en lab, que separa luz y color):
#   media (3), desviacion (3), diferencia entre izquierda y derecha (3), diferencia entre arriba y abajo (3)
# las diferencias van en valor absoluto, asi la foto en espejo da los mismos numeros
def getLocalColor(img:cv.Mat, mask:cv.Mat) -> np.ndarray:
    lab = cv.cvtColor(img.astype(np.float32) / 255, cv.COLOR_BGR2Lab)
    half = PARCHE // 2
    patches = []
    for y in range(0, SIZE - PARCHE + 1, PASO):
        for x in range(0, SIZE - PARCHE + 1, PASO):
            if mask[y:y+PARCHE, x:x+PARCHE].mean() < 0.6 * 255:
                continue
            p = lab[y:y+PARCHE, x:x+PARCHE]
            left, right = p[:, :half].mean(axis=(0, 1)), p[:, half:].mean(axis=(0, 1))
            top, bottom = p[:half].mean(axis=(0, 1)), p[half:].mean(axis=(0, 1))
            patches.append(np.concatenate([p.mean(axis=(0, 1)), p.std(axis=(0, 1)), np.abs(left - right), np.abs(top - bottom)]))
    if len(patches) == 0:
        return np.zeros((0, 12), np.float32)
    return np.array(patches, np.float32)


# fisher vectors del color local (extra, igual que el de sift pero sin pca porque ya son solo 12 numeros)
#   1. estandarizar los 12 numeros de cada parche (con la media y desviacion de las fotos de entrenamiento)
#   2. cada parche reparte su peso entre las K_GMM gaussianas
#   3. por cada gaussiana guardamos cuanto se aleja la media de los parches de su centro y cuanto cambia su varianza
#   4. raiz cuadrada con signo (asi no mandan los valores muy grandes)
def getFisherColor(patches:np.ndarray, fc:dict) -> np.ndarray:
    size = 2 * K_GMM * 12
    if len(patches) < 2:
        return np.zeros(size, np.float32)
    x = (patches - fc["mean"]) / fc["std"]
    w, mu, var = fc["weights"], fc["means"], fc["variances"]
    maha = (x*x) @ (1/var).T - 2 * x @ (mu/var).T + (mu*mu/var).sum(axis=1)[np.newaxis, :]
    logp = -0.5 * (maha + np.log(var).sum(axis=1)[np.newaxis, :]) + np.log(w)[np.newaxis, :]
    logp = logp - logp.max(axis=1, keepdims=True)
    gamma = np.exp(logp)
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    n = len(x)
    s0 = gamma.sum(axis=0)
    s1 = gamma.T @ x
    s2 = gamma.T @ (x*x)
    g_mu = (s1 - s0[:, np.newaxis]*mu) / (np.sqrt(var) * np.sqrt(w)[:, np.newaxis] * n)
    g_var = (s2 - 2*mu*s1 + (mu*mu - var)*s0[:, np.newaxis]) / (var * np.sqrt(2*w)[:, np.newaxis] * n)
    fv = np.concatenate([g_mu.flatten(), g_var.flatten()])
    return (np.sign(fv) * np.sqrt(np.abs(fv))).astype(np.float32)


# cargar el vocabulario y los datos de fisher del disco (solo la primera vez)
def loadVocab() -> tuple:
    global vocab, fisher, fisher_color
    if vocab is None:
        vocab = np.load(vocabulario)
        d = np.load(fisher_modelo)
        fisher = {key: d[key] for key in d.files}
        d = np.load(fisher_color_modelo)
        fisher_color = {key: d[key] for key in d.files}
    return vocab, fisher, fisher_color


# todos los descriptores de una foto (devuelve un diccionario)
# con flip=True calcula tambien los de la foto en espejo (se usan como fotos extra al entrenar)
def describeImage(img:cv.Mat, mask:cv.Mat, flip:bool=False) -> dict:
    vocab, fisher, fisher_color = loadVocab()
    d = {}
    d["color"] = getColorHist(img, mask)
    d["hog"] = getHOG(img)
    d["lbp"] = getLBP(img, mask, 1)
    d["lbp2"] = getLBP(img, mask, 2)
    des = getDenseSIFT(img, mask)
    d["bow"] = getBoW(des, vocab)
    d["fisher"] = getFisher(des, fisher)
    d["lcolor"] = getFisherColor(getLocalColor(img, mask), fisher_color)
    if flip:
        img_flip = cv.flip(img, 1)
        mask_flip = cv.flip(mask, 1)
        d["hog_flip"] = getHOG(img_flip)
        d["lbp_flip"] = getLBP(img_flip, mask_flip, 1)
        d["lbp2_flip"] = getLBP(img_flip, mask_flip, 2)
        des = getDenseSIFT(img_flip, mask_flip)
        d["bow_flip"] = getBoW(des, vocab)
        d["fisher_flip"] = getFisher(des, fisher)
    return d


# coger 32 descriptores sift de una foto, repartidos por todo el perro (la usa el Pool)
def sampleDescriptors(name:str) -> np.ndarray:
    img = cv.imread(carpeta + "/" + name + ".png")
    mask = cv.imread(carpeta_mask + "/" + name + ".png", cv.IMREAD_GRAYSCALE)
    des = getDenseSIFT(img, mask)
    if len(des) > 32:
        des = des[::len(des)//32][:32]    # uno de cada tantos (con todos tardaria muchisimo)
    return des


# paso 1: descriptores sift de una de cada 4 fotos de entrenamiento (se guardan para los pasos 2 y 3)
# se hace en trozos de 200 fotos, cada uno en su fichero, para poder parar y seguir
# devuelve True si ya estan todos los trozos
def stepSamples(names_train:list, t_end:float) -> bool:
    parts = muestras[:-4] + "_partes"
    os.makedirs(parts, exist_ok=True)
    chosen = names_train[::4]
    size = 200
    with Pool(os.cpu_count()) as pool:
        for i in range(0, len(chosen), size):
            part = parts + "/parte_" + str(i // size) + ".npy"
            if os.path.exists(part):
                continue
            if t_end is not None and time.time() > t_end:
                print("se acabo el tiempo, vuelve a ejecutar para seguir", flush=True)
                return False
            des = pool.map(sampleDescriptors, chosen[i:i+size], chunksize=4)
            np.save(part + ".tmp.npy", np.vstack(des).astype(np.float32))
            os.replace(part + ".tmp.npy", part)
    des = np.vstack([np.load(parts + "/" + f) for f in sorted(os.listdir(parts)) if not f.endswith(".tmp.npy")])
    np.save(muestras, des)
    print("descriptores sift para los vocabularios:", des.shape, flush=True)
    return True


# paso 2: vocabulario del bag of words con k-means (tema 5)
# k-means a mano (con numpy) para poder parar y seguir: guardamos los centros en cada vuelta
#   1. elegimos K descriptores al azar como centros
#   2. cada descriptor se asigna al centro mas cercano
#   3. cada centro pasa a ser la media de sus descriptores
#   4. repetimos 2 y 3 varias veces
# devuelve True si ya ha terminado
ITER = 20     # vueltas de k-means
def stepVocabulary(t_end:float) -> bool:
    estado = muestras[:-4] + "_kmeans.npz"
    des = np.load(muestras)
    if os.path.exists(estado):
        e = np.load(estado)
        center, it = e["center"], int(e["it"])
    else:
        rng = np.random.RandomState(0)
        center, it = des[rng.choice(len(des), K, replace=False)].copy(), 0
    des_sq = (des * des).sum(axis=1)[:, np.newaxis]
    while it < ITER:
        if t_end is not None and time.time() > t_end:
            print("k-means parado en la vuelta", it, "- vuelve a ejecutar para seguir", flush=True)
            return False
        # distancia de cada descriptor a cada centro (en trozos para no gastar memoria)
        label = np.zeros(len(des), np.int32)
        for i in range(0, len(des), 20000):
            d = des_sq[i:i+20000] - 2 * des[i:i+20000] @ center.T + (center * center).sum(axis=1)[np.newaxis, :]
            label[i:i+20000] = d.argmin(axis=1)
        for k in range(K):
            if (label == k).any():
                center[k] = des[label == k].mean(axis=0)
        it += 1
        np.savez(estado + ".tmp.npz", center=center, it=it)
        os.replace(estado + ".tmp.npz", estado)
        print("k-means vuelta", it, flush=True)
    np.save(vocabulario, center)
    print("vocabulario hecho:", center.shape, flush=True)
    return True


# paso 3: pca y gaussianas para los fisher vectors (extra)
def stepFisher() -> None:
    des = np.load(muestras)
    x = np.sqrt(des / np.maximum(des.sum(axis=1, keepdims=True), 1e-7))
    pca = PCA(n_components=D_PCA, random_state=0).fit(x)
    gmm = GaussianMixture(n_components=K_GMM, covariance_type="diag", max_iter=40, random_state=0)
    gmm.fit(pca.transform(x))
    np.savez(fisher_modelo, mean_pca=pca.mean_, components=pca.components_,
             weights=gmm.weights_, means=gmm.means_, variances=gmm.covariances_)
    print("fisher hecho:", gmm.means_.shape, flush=True)


# coger los parches de color local de una foto (unos 50, repartidos por todo el perro; la usa el Pool)
def samplePatches(name:str) -> np.ndarray:
    img = cv.imread(carpeta + "/" + name + ".png")
    mask = cv.imread(carpeta_mask + "/" + name + ".png", cv.IMREAD_GRAYSCALE)
    patches = getLocalColor(img, mask)
    if len(patches) > 50:
        patches = patches[::len(patches)//50][:50]    # uno de cada tantos
    return patches


# paso 3b: gaussianas para los fisher vectors del color local (extra); igual que el paso 3 pero con los parches
def stepFisherColor(names_train:list) -> None:
    with Pool(os.cpu_count()) as pool:
        parts = pool.map(samplePatches, names_train[::4], chunksize=20)
    x = np.vstack(parts)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    gmm = GaussianMixture(n_components=K_GMM, covariance_type="diag", max_iter=40, random_state=0, reg_covar=1e-3)
    gmm.fit((x - mean) / std)
    np.savez(fisher_color_modelo, mean=mean, std=std, weights=gmm.weights_, means=gmm.means_, variances=gmm.covariances_)
    print("fisher del color local hecho:", gmm.means_.shape, "con", len(x), "parches", flush=True)


# anadir el color local a un .npz que ya estaba hecho (la usa el Pool)
# task = (nombre de la foto, carpeta de fotos, carpeta de mascaras, carpeta de los .npz)
def addColorFile(task:tuple) -> int:
    name, folder, folder_mask, out = task
    path = out + "/" + name + ".npz"
    d = np.load(path)
    if "lcolor" in d.files:
        return 0
    d = {key: d[key] for key in d.files}
    img = cv.imread(folder + "/" + name + ".png")
    mask = cv.imread(folder_mask + "/" + name + ".png", cv.IMREAD_GRAYSCALE)
    d["lcolor"] = getFisherColor(getLocalColor(img, mask), loadVocab()[2])
    np.savez(path + ".tmp.npz", **d)
    os.replace(path + ".tmp.npz", path)
    return 1


# anadir el color local a todos los .npz de una carpeta (se salta los que ya lo tienen)
def addColorFolder(folder:str, folder_mask:str, out:str, t_end:float) -> None:
    names = [n[:-4] for n in sorted(os.listdir(out)) if n.endswith(".npz") and not n.endswith(".tmp.npz")]
    tasks = [(n, folder, folder_mask, out) for n in names]
    print(out, "-> comprobando el color local de", len(tasks), "fotos", flush=True)
    batch = 40 * os.cpu_count()
    new = 0
    with Pool(os.cpu_count()) as pool:
        for i in range(0, len(tasks), batch):
            if t_end is not None and time.time() > t_end:
                print("se acabo el tiempo, vuelve a ejecutar para seguir (llevo", i, "fotos revisadas,", new, "nuevas)", flush=True)
                return
            new += sum(pool.map(addColorFile, tasks[i:i+batch], chunksize=8))
            if (i // batch) % 25 == 0:
                print(i, "revisadas,", new, "nuevas", flush=True)


# describir una foto del disco y guardar su .npz (la usa el Pool)
# task = (nombre de la foto, carpeta de fotos, carpeta de mascaras, carpeta de salida, con espejo)
def describeFile(task:tuple) -> str:
    name, folder, folder_mask, out, flip = task
    img = cv.imread(folder + "/" + name + ".png")
    mask = cv.imread(folder_mask + "/" + name + ".png", cv.IMREAD_GRAYSCALE)
    d = describeImage(img, mask, flip)
    final = out + "/" + name + ".npz"
    np.savez(final + ".tmp.npz", **d)
    os.replace(final + ".tmp.npz", final)
    return name


# describir las fotos de una carpeta (saltamos las que ya estan hechas)
# flips = nombres de las fotos que tambien necesitan los descriptores en espejo
def describeFolder(folder:str, folder_mask:str, out:str, flips:set, t_end:float) -> None:
    os.makedirs(out, exist_ok=True)
    done = set(n[:-4] for n in os.listdir(out) if n.endswith(".npz"))
    names = [n[:-4] for n in sorted(os.listdir(folder)) if n.endswith(".png")]
    tasks = [(n, folder, folder_mask, out, n in flips) for n in names if n not in done]
    print(folder, "-> fotos por describir:", len(tasks), flush=True)
    batch = 8 * os.cpu_count()
    with Pool(os.cpu_count()) as pool:
        for i in range(0, len(tasks), batch):
            if t_end is not None and time.time() > t_end:
                print("se acabo el tiempo, vuelve a ejecutar para seguir", flush=True)
                return
            pool.map(describeFile, tasks[i:i+batch])
            if (i // batch) % 25 == 0:
                print(i, "hechas", flush=True)


def main():
    cv.setNumThreads(1)
    t_end = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(os.path.dirname(vocabulario), exist_ok=True)
    names, breeds, labels, splits = listDataset(carpeta)
    names_train = [n for n, s in zip(names, splits) if s == "train"]
    print("fotos:", len(names), "| entrenamiento:", len(names_train), "| razas:", len(breeds), flush=True)

    # pasos 1, 2, 3 y 3b: vocabularios (solo con fotos de entrenamiento; se saltan si ya estan hechos)
    if not os.path.exists(muestras):
        if not stepSamples(names_train, t_end):
            return
    if not os.path.exists(vocabulario):
        if not stepVocabulary(t_end):
            return
    if not os.path.exists(fisher_modelo):
        stepFisher()
    if not os.path.exists(fisher_color_modelo):
        stepFisherColor(names_train)
    if t_end is not None and time.time() > t_end:
        return

    # paso 4: descriptores de todas las fotos del dataset (las de entrenamiento tambien en espejo)
    describeFolder(carpeta, carpeta_mask, salida, set(names_train), t_end)
    addColorFolder(carpeta, carpeta_mask, salida, t_end)       # (el color local se anadio despues: lo ponen las fotos que ya estaban hechas)

    # paso 5: descriptores de las fotos descartadas (solo para probar luego que pasa con ellas)
    if os.path.isdir(descartadas):
        describeFolder(descartadas, descartadas_mask, salida_descartadas, set(), t_end)
        addColorFolder(descartadas, descartadas_mask, salida_descartadas, t_end)
    print("terminado", flush=True)


if __name__ == "__main__":
    main()
