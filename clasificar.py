import numpy as np
import os
import sys
import time
import pickle
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import cho_factor, cho_solve

# ------------------------------------------------------------
# clasificar: aprender la raza de un perro a partir de sus descriptores
#
# las fotos se reparten con la division oficial de stanford dogs:
#   train -> 100 fotos por raza (para aprender)
#   test  -> el resto (solo para medir el resultado final)
#
# se puede parar y volver a ejecutar (guarda lo hecho en disco):
#   python clasificar.py cargar [segundos]  -> junta los descriptores de todas las fotos en matrices
#   python clasificar.py final              -> entrena, mide en test y guarda el modelo
#
# clasificador final: lda con shrinkage (no visto en clase)
# las pruebas con otros clasificadores (knn y svm de clase, ridge) y con la confianza
# estan en la carpeta experimentos
# ------------------------------------------------------------

carpeta = "datos/perros_256"                  # fotos recortadas
carpeta_desc = "datos/descriptores"           # un .npz por foto
carpeta_desc_descartadas = "datos/descriptores_descartadas"
descartadas = "datos/descartadas"             # fotos quitadas del dataset (desastres)
matrices = "datos/matrices"                   # descriptores juntos en matrices
resultados = "resultados"                     # tablas
modelo = "modelo"
lista = "datos/lista_fotos.csv"
CLAVES = ["color", "hog", "lbp", "lbp2", "bow", "fisher", "lcolor"]    # todos los descriptores
RAIZ = ["color", "lbp", "lbp2", "bow"]                      # histogramas: se les hace la raiz cuadrada
ALFA = 0.7                    # cuanto se mezcla la covarianza con su diagonal en el lda (elegido con validacion)


# ---------- lista de fotos ----------

# lista de fotos: nombre -> (raza, train o test)
def readList() -> dict:
    info = {}
    for line in open(lista).read().split("\n")[1:]:
        if line:
            name, breed, split = line.split(",")[:3]
            info[name] = (breed, split)
    return info


# de donde viene cada foto (stanford u oxford)
def readSources() -> dict:
    source = {}
    for line in open(lista).read().split("\n")[1:]:
        if line:
            cols = line.split(",")
            source[cols[0]] = cols[3]
    return source


# fotos del dataset, razas (las 120), raza de cada foto (numero) y grupo (train o test)
def listDataset() -> tuple:
    info = readList()
    names = [n[:-4] for n in sorted(os.listdir(carpeta)) if n.endswith(".png") and n[:-4] in info]
    breeds = sorted(set(breed for breed, split in info.values()))
    y = np.array([breeds.index(info[n][0]) for n in names])
    split = np.array([info[n][1] for n in names])
    return names, breeds, y, split


# ---------- paso cargar ----------

# leer los descriptores de una foto
def readNpz(name:str) -> dict:
    d = np.load(carpeta_desc + "/" + name + ".npz")
    return {key: d[key] for key in d.files}


# paso cargar: juntar los 20 mil ficheros .npz en una matriz por descriptor
# se hace en trozos de 1500 fotos (cada uno en su fichero) para poder parar y seguir
def stepLoad(t_end:float) -> None:
    # si ya estan todas las matrices no hay nada que hacer
    if all(os.path.exists(matrices + "/" + k + ".npy") for k in CLAVES + ["y", "split"]):
        print("las matrices ya estan hechas", flush=True)
        return
    names, breeds, y, split = listDataset()
    os.makedirs(matrices + "/trozos", exist_ok=True)
    size = 1500
    with ThreadPoolExecutor(16) as ex:
        for i in range(0, len(names), size):
            path = matrices + "/trozos/trozo_" + str(i // size) + ".npz"
            if os.path.exists(path):
                continue
            if time.time() > t_end:
                print("se acabo el tiempo, vuelve a ejecutar para seguir", flush=True)
                return
            rows = list(ex.map(readNpz, names[i:i+size]))
            out = {}
            # el color y el color local no tienen espejo (dan lo mismo con la foto en espejo)
            for key in CLAVES + [c + "_flip" for c in CLAVES if c not in ("color",)]:
                if key.endswith("_flip") and not any(key in r for r in rows):
                    continue
                # las fotos de test no tienen espejo: se rellenan con ceros (no se usan)
                out[key] = np.array([r[key] if key in r else np.zeros_like(r[key[:-5]]) for r in rows], np.float32)
            np.savez(path + ".tmp.npz", **out)
            os.replace(path + ".tmp.npz", path)
            print("trozo", i // size, "hecho", flush=True)
    # juntar los trozos: una matriz <descriptor>.npy y otra <descriptor>_flip.npy con los espejos (solo valen en train)
    parts = [np.load(matrices + "/trozos/trozo_" + str(j) + ".npz") for j in range(-(-len(names) // size))]
    for key in parts[0].files:
        if os.path.exists(matrices + "/" + key + ".npy"):
            continue
        np.save(matrices + "/" + key + "_tmp.npy", np.vstack([p[key] for p in parts]))
        os.replace(matrices + "/" + key + "_tmp.npy", matrices + "/" + key + ".npy")
    np.save(matrices + "/y.npy", y)
    np.save(matrices + "/split.npy", split)
    print("matrices listas:", len(names), "fotos", flush=True)


# ---------- matrices de descriptores ----------

# cargar las matrices de descriptores (una por descriptor, una fila por foto)
def loadData() -> dict:
    data = {}
    for key in CLAVES:
        data[key] = np.load(matrices + "/" + key + ".npy")
        if os.path.exists(matrices + "/" + key + "_flip.npy"):
            data[key + "_flip"] = np.load(matrices + "/" + key + "_flip.npy")
    return data


# coger un descriptor (normal o de la foto en espejo) y prepararlo
def getBlock(data:dict, key:str, idx:np.ndarray, flip:bool) -> np.ndarray:
    x = data[key + "_flip"][idx] if flip and key + "_flip" in data else data[key][idx]
    if key in RAIZ:
        x = np.sqrt(np.maximum(x, 0))    # raiz cuadrada: asi los valores muy grandes del histograma no mandan tanto
    return x


# media y desviacion de cada descriptor, calculadas solo con las fotos de entrenamiento
def fitScaler(data:dict, idx:np.ndarray, keys:list) -> dict:
    scaler = {}
    for key in keys:
        x = getBlock(data, key, idx, False)
        scaler[key] = (x.mean(axis=0), x.std(axis=0) + 1e-6)
    return scaler


# juntar varios descriptores en una sola matriz
#   1. estandarizar: restar la media y dividir entre la desviacion (todos en la misma escala)
#   2. dividir entre la raiz del tamano: asi un descriptor largo no pesa mas que uno corto
def buildMatrix(data:dict, idx:np.ndarray, keys:list, scaler:dict, flip:bool=False) -> np.ndarray:
    blocks = []
    for key in keys:
        x = getBlock(data, key, idx, flip)
        mean, std = scaler[key]
        blocks.append(((x - mean) / std / np.sqrt(x.shape[1])).astype(np.float32))
    return np.hstack(blocks)


# datos de entrenamiento: las fotos y ademas sus espejos (el doble de fotos)
def getTrainSet(data:dict, y:np.ndarray, idx:np.ndarray, keys:list, scaler:dict, flip:bool=True) -> tuple:
    X = buildMatrix(data, idx, keys, scaler)
    if not flip:
        return X, y[idx]
    X = np.vstack([X, buildMatrix(data, idx, keys, scaler, True)])
    return X, np.concatenate([y[idx], y[idx]])


# ---------- clasificador (lda) ----------

# lda con shrinkage (no visto en clase)
#   1. la foto "media" de cada raza (en el espacio de los descriptores)
#   2. la covarianza: cuanto se mueven las fotos de una misma raza alrededor de su media, y que descriptores se mueven juntos
#   3. la nota de una foto para una raza = lo cerca que esta de la media de esa raza, midiendo con esa covarianza
# con tantos numeros por foto la covarianza sale muy mala, asi que la mezclamos con su diagonal (shrinkage):
#   covarianza_final = (1 - alfa) * covarianza + alfa * su diagonal
def trainLDA(X:np.ndarray, y:np.ndarray, n_breeds:int, alpha:float=ALFA) -> tuple:
    mu = X.mean(axis=0)
    Xc = X - mu
    means = np.array([Xc[y == b].mean(axis=0) for b in range(n_breeds)])
    for b in range(n_breeds):
        Xc[y == b] -= means[b]                  # a cada foto le restamos la media de su raza
    S = Xc.T @ Xc / len(X)
    del Xc
    diag = np.diag(S).copy()
    S *= 1 - alpha
    S[np.diag_indices_from(S)] += alpha * diag + 1e-3 * diag.mean()
    W = cho_solve(cho_factor(S, overwrite_a=True, check_finite=False), means.T, check_finite=False)
    bias = -0.5 * (means * W.T).sum(axis=1)
    return W.astype(np.float32), bias.astype(np.float32), mu


# nota de cada raza para cada foto
def scoreLDA(X:np.ndarray, W:np.ndarray, bias:np.ndarray, mu:np.ndarray) -> np.ndarray:
    return (X - mu) @ W + bias


# ---------- medidas ----------

# acierto top-1 (la primera raza es la buena) y top-3 (la buena esta entre las 3 primeras)
def getAccuracy(S:np.ndarray, y_true:np.ndarray) -> tuple:
    order = np.argsort(-S, axis=1)[:, :3]
    return (order[:, 0] == y_true).mean(), (order == y_true[:, np.newaxis]).any(axis=1).mean()


# guardar una linea de resultados en un csv (se puede repetir: no pasa nada)
def saveRow(file:str, row:list) -> None:
    os.makedirs(resultados, exist_ok=True)
    f = open(resultados + "/" + file, "a")
    f.write(",".join(str(v) for v in row) + "\n")
    f.close()


# descriptores de las fotos descartadas / dudosas con su raza (para medir como les va)
def loadExtra(names:list, folder:str, breeds:list, info:dict) -> tuple:
    data, keep = {}, []
    for n in names:
        path = folder + "/" + n + ".npz"
        if os.path.exists(path) and n in info:
            d = np.load(path)
            for key in CLAVES:
                data.setdefault(key, []).append(d[key])
            keep.append(n)
    for key in data:
        data[key] = np.array(data[key], np.float32)
    return data, keep


# ---------- paso final ----------

# acierto por raza y las razas que mas se confunden (guarda dos csv)
def saveBreedTables(S:np.ndarray, y_test:np.ndarray, breeds:list) -> None:
    order = np.argsort(-S, axis=1)[:, :3]
    f = open(resultados + "/acierto_por_raza.csv", "w")
    f.write("raza,fotos_test,top1,top3\n")
    for b in range(120):
        m = y_test == b
        f.write("%s,%d,%.3f,%.3f\n" % (breeds[b], m.sum(), (order[m, 0] == b).mean(), (order[m] == b).any(axis=1).mean()))
    f.close()
    # cuantas veces se dice una raza (raza_dicha) cuando era otra (raza_real)
    pairs = {}
    for i in range(len(y_test)):
        if order[i, 0] != y_test[i]:
            pairs[(y_test[i], order[i, 0])] = pairs.get((y_test[i], order[i, 0]), 0) + 1
    f = open(resultados + "/confusiones.csv", "w")
    f.write("raza_real,raza_dicha,fotos\n")
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:30]:
        f.write("%s,%s,%d\n" % (breeds[a], breeds[b], n))
    f.close()


# acierto de un grupo de fotos del test (mask = que fotos del test entran en el grupo)
def saveGroup(name:str, S:np.ndarray, y_test:np.ndarray, mask:np.ndarray) -> None:
    top1, top3 = getAccuracy(S[mask], y_test[mask])
    saveRow("final.csv", [name, mask.sum(), "%.4f" % top1, "%.4f" % top3])
    print(name, ":", mask.sum(), "fotos top1 %.3f top3 %.3f" % (top1, top3), flush=True)


# acierto en las fotos descartadas (desastres) que son de test
def saveDiscarded(breeds:list, scaler:dict, W:np.ndarray, bias:np.ndarray, mu:np.ndarray) -> None:
    info = readList()
    names = [n[:-4] for n in os.listdir(descartadas) if n.endswith(".png")]
    extra, keep = loadExtra(names, carpeta_desc_descartadas, breeds, info)
    keep_test = np.array([i for i, n in enumerate(keep) if info[n][1] == "test"])
    if len(keep_test) == 0:
        return
    y_d = np.array([breeds.index(info[keep[i]][0]) for i in keep_test])
    S = scoreLDA(buildMatrix(extra, keep_test, CLAVES, scaler), W, bias, mu)
    top1, top3 = getAccuracy(S, y_d)
    saveRow("final.csv", ["test_descartadas", len(y_d), "%.4f" % top1, "%.4f" % top3])
    print("test_descartadas :", len(y_d), "fotos top1 %.3f top3 %.3f" % (top1, top3), flush=True)


# entrenar con todo train, medir en test (solo una vez) y guardar el modelo
def stepFinal() -> None:
    os.makedirs(resultados, exist_ok=True)
    f = open(resultados + "/final.csv", "w")    # empezamos el fichero de cero
    f.write("conjunto,fotos,top1,top3\n")
    f.close()
    y, split = np.load(matrices + "/y.npy"), np.load(matrices + "/split.npy")
    idx_train, idx_test = np.where(split == "train")[0], np.where(split == "test")[0]
    names, breeds, y_all, split_all = listDataset()
    data = loadData()
    y_test = y[idx_test]

    # 1. entrenar con las fotos de train y sus espejos
    scaler = fitScaler(data, idx_train, CLAVES)
    X, y_train = getTrainSet(data, y, idx_train, CLAVES, scaler)
    W, bias, mu = trainLDA(X, y_train, 120)
    del X

    # 2. medir en test
    S = scoreLDA(buildMatrix(data, idx_test, CLAVES, scaler), W, bias, mu)
    top1, top3 = getAccuracy(S, y_test)
    saveRow("final.csv", ["test", len(idx_test), "%.4f" % top1, "%.4f" % top3])
    print("TEST (", len(idx_test), "fotos ): top1 %.4f top3 %.4f" % (top1, top3), flush=True)

    # 3. acierto por raza y confusiones
    saveBreedTables(S, y_test, breeds)

    # 4. solo las fotos de stanford (las 3 razas de oxford suben el acierto porque sus fotos tienen otro estilo)
    source = readSources()
    saveGroup("test_solo_stanford", S, y_test, np.array([source[names[i]] == "stanford" for i in idx_test]))

    # 5. fotos dudosas del test (el recorte no se veia seguro) y fotos descartadas
    dudosas = set(l[:-4] for l in open(carpeta + "/dudosas.txt").read().split("\n") if l)
    is_doubt = np.array([names[i] in dudosas for i in idx_test])
    if is_doubt.any():
        saveGroup("test_dudosas", S, y_test, is_doubt)
        saveGroup("test_no_dudosas", S, y_test, ~is_doubt)
    saveDiscarded(breeds, scaler, W, bias, mu)

    # 6. guardar el modelo para reconocer_raza.py
    os.makedirs(modelo, exist_ok=True)
    f = open(modelo + "/modelo.pkl", "wb")
    pickle.dump({"breeds": breeds, "keys": CLAVES, "scaler": scaler, "W": W, "bias": bias, "mu": mu}, f)
    f.close()
    print("modelo guardado", flush=True)


def main():
    paso = sys.argv[1] if len(sys.argv) > 1 else "cargar"
    t_end = time.time() + (float(sys.argv[2]) if len(sys.argv) > 2 else 150)
    if paso == "cargar":
        stepLoad(t_end)
    if paso == "final":
        stepFinal()


if __name__ == "__main__":
    main()
