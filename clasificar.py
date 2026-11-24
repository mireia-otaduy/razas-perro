import cv2 as cv
import numpy as np
import os
import sys
import time
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.svm import SVC

# ------------------------------------------------------------
# clasificar (version 2): aprender a decir la raza a partir de los descriptores
#
# las fotos se reparten asi (division oficial de stanford dogs):
#   train -> 100 fotos por raza (para aprender)
#   test  -> el resto (solo para el resultado final, no se toca hasta el final)
# dentro de train separamos un 20% de "validacion" para elegir descriptores y parametros
#
# como cada paso tarda, se guarda todo en disco y se puede parar y seguir:
#   python clasificar.py cargar       -> junta los descriptores de las fotos en matrices
#   python clasificar.py descriptores -> que descriptores funcionan mejor (validacion)
#   python clasificar.py comparar     -> ridge contra svm y knn (validacion)
#   python clasificar.py final        -> resultado en test, umbral de confianza y modelo guardado
#   python clasificar.py razas        -> como baja el acierto al subir el numero de razas
#
# clasificador final: regresion lineal con regularizacion (ridge) sobre todos los descriptores
#   (NO visto en clase, es un clasificador lineal muy simple: aprende una nota por raza)
# lo comparamos con los de clase: knn (practica 6.1) y svm con kernel rbf (practica 6.2)
# ------------------------------------------------------------

carpeta = "datos/perros_256"
carpeta_desc = "datos/descriptores"
descartadas = "datos/descartadas"            # fotos quitadas del dataset (desastres)
matrices = "datos/matrices"        # descriptores juntos en matrices
resultados = "resultados"          # tablas y graficas
modelo = "modelo"
lista = "datos/lista_fotos.csv"
CLAVES = ["color", "hog", "lbp", "lbp2", "bow", "fisher"]    # todos los descriptores
RAIZ = ["color", "lbp", "lbp2", "bow"]                      # histogramas: se les hace la raiz cuadrada


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


# leer los descriptores de una foto
def readNpz(name:str) -> dict:
    d = np.load(carpeta_desc + "/" + name + ".npz")
    return {key: d[key] for key in d.files}


# paso cargar: juntar los 20 mil ficheros .npz en una matriz por descriptor
# se hace en trozos de 1500 fotos (cada uno en su fichero) para poder parar y seguir
def stepLoad(t_end:float) -> None:
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
            for key in CLAVES + [c + "_flip" for c in CLAVES if c not in ("color",)]:
                if key.endswith("_flip") and not any(key in r for r in rows):
                    continue
                out[key] = np.array([r[key] if key in r else np.zeros_like(r[key[:-5]]) for r in rows], np.float32)
            np.savez(path + ".tmp.npz", **out)
            os.replace(path + ".tmp.npz", path)
            print("trozo", i // size, "hecho", flush=True)
    # juntar los trozos: all_<clave>.npy (todas las fotos) y flip_<clave>.npy (espejo; solo vale en train)
    parts = [np.load(matrices + "/trozos/trozo_" + str(j) + ".npz") for j in range(-(-len(names) // size))]
    for key in parts[0].files:
        if os.path.exists(matrices + "/" + key + ".npy"):
            continue
        np.save(matrices + "/" + key + "_tmp.npy", np.vstack([p[key] for p in parts]))
        os.replace(matrices + "/" + key + "_tmp.npy", matrices + "/" + key + ".npy")
    np.save(matrices + "/y.npy", y)
    np.save(matrices + "/split.npy", split)
    print("matrices listas:", len(names), "fotos", flush=True)


LAMBDA = 8.0                  # fuerza de la regularizacion del ridge (elegida con validacion)
RESULTADOS = resultados


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


# validacion: 20% de las fotos de train de cada raza (el resto se llama fit)
def splitValidation(y:np.ndarray, idx_train:np.ndarray) -> tuple:
    rng = np.random.RandomState(0)
    fit, val = [], []
    for b in np.unique(y):
        ii = idx_train[y[idx_train] == b].copy()
        rng.shuffle(ii)
        n = int(round(len(ii) * 0.2))
        val += list(ii[:n])
        fit += list(ii[n:])
    return np.array(sorted(fit)), np.array(sorted(val))


# ---------- clasificadores ----------

# ridge: busca una nota lineal por raza (minimos cuadrados con una penalizacion para no pasarse)
def trainRidge(X:np.ndarray, y:np.ndarray, n_breeds:int, lam:float=LAMBDA) -> tuple:
    mu = X.mean(axis=0)
    Xc = X - mu
    T = np.zeros((len(y), n_breeds), np.float32)
    T[np.arange(len(y)), y] = 1
    T -= T.mean(axis=0)
    A = Xc.T @ Xc
    A[np.diag_indices_from(A)] += lam * len(X) / X.shape[1]
    W = np.linalg.solve(A.astype(np.float64), (Xc.T @ T).astype(np.float64))
    return W.astype(np.float32), mu


# nota de cada raza para cada foto
def scoreRidge(X:np.ndarray, W:np.ndarray, mu:np.ndarray) -> np.ndarray:
    return (X - mu) @ W


# kernel rbf calculado de golpe con numpy (el mismo kernel de la practica 6.2, pero mucho mas rapido)
def rbfKernel(A:np.ndarray, B:np.ndarray, gamma:float) -> np.ndarray:
    out = np.empty((len(A), len(B)), np.float32)
    b2 = (B*B).sum(axis=1)[np.newaxis, :]
    for i in range(0, len(A), 2000):
        a = A[i:i+2000]
        d2 = (a*a).sum(axis=1)[:, np.newaxis] + b2 - 2 * a @ B.T
        out[i:i+2000] = np.exp(-gamma * np.maximum(d2, 0))
    return out


# svm con kernel rbf (practica 6.2); devuelve la nota de cada raza para cada foto
def scoreSVM(X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, gamma:float, C:float=10) -> np.ndarray:
    svm = SVC(C=C, kernel="precomputed", decision_function_shape="ovr")
    svm.fit(rbfKernel(X_train, X_train, gamma).astype(np.float64), y_train)
    return svm.decision_function(rbfKernel(X_test, X_train, gamma))


# knn de opencv (practica 6.1): devuelve los votos de cada raza entre los k vecinos
def scoreKNN(X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, k:int, n_breeds:int) -> np.ndarray:
    knn = cv.ml.KNearest_create()
    knn.train(X_train.astype(np.float32), cv.ml.ROW_SAMPLE, y_train.astype(np.float32))
    ret, result, neighbours, dist = knn.findNearest(X_test.astype(np.float32), k)
    votes = np.zeros((len(X_test), n_breeds), np.float32)
    for i in range(len(X_test)):
        for n in neighbours[i]:
            votes[i, int(n)] += 1
    return votes


# ---------- medidas ----------

# acierto top-1 (la primera raza es la buena) y top-3 (la buena esta entre las 3 primeras)
def getAccuracy(S:np.ndarray, y_true:np.ndarray) -> tuple:
    order = np.argsort(-S, axis=1)[:, :3]
    return (order[:, 0] == y_true).mean(), (order == y_true[:, np.newaxis]).any(axis=1).mean()


# guardar una linea de resultados en un csv (se puede repetir: no pasa nada)
def saveRow(file:str, row:list) -> None:
    os.makedirs(RESULTADOS, exist_ok=True)
    f = open(RESULTADOS + "/" + file, "a")
    f.write(",".join(str(v) for v in row) + "\n")
    f.close()


# ---------- pasos ----------

# paso descriptores: que descriptor funciona mejor (ridge, entrenando con fit y midiendo en val)
def stepDescriptors(data:dict, y:np.ndarray, fit:np.ndarray, val:np.ndarray, t_end:float) -> None:
    combos = [[k] for k in CLAVES] + [CLAVES] + [[k for k in CLAVES if k != q] for q in CLAVES]
    done = set()
    if os.path.exists(RESULTADOS + "/descriptores.csv"):
        done = set(l.split(",")[0] for l in open(RESULTADOS + "/descriptores.csv").read().split("\n") if l)
    for keys in combos:
        name = "+".join(keys)
        if name in done:
            continue
        if time.time() > t_end:
            print("se acabo el tiempo, vuelve a ejecutar para seguir", flush=True)
            return
        scaler = fitScaler(data, fit, keys)
        X, y_train = getTrainSet(data, y, fit, keys, scaler)
        W, mu = trainRidge(X, y_train, 120)
        S = scoreRidge(buildMatrix(data, val, keys, scaler), W, mu)
        top1, top3 = getAccuracy(S, y[val])
        saveRow("descriptores.csv", [name, "%.4f" % top1, "%.4f" % top3])
        print(name, "top1 %.3f top3 %.3f" % (top1, top3), flush=True)


# paso comparar: ridge contra los clasificadores de clase (knn y svm), todos con los mismos descriptores
def stepCompare(data:dict, y:np.ndarray, fit:np.ndarray, val:np.ndarray, t_end:float) -> None:
    scaler = fitScaler(data, fit, CLAVES)
    X_fit, y_fit = getTrainSet(data, y, fit, CLAVES, scaler, False)      # sin espejos
    X_flip, y_flip = getTrainSet(data, y, fit, CLAVES, scaler, True)     # con espejos
    X_val = buildMatrix(data, val, CLAVES, scaler)
    runs = []
    for k in (5, 15, 30):
        runs.append(("knn k=%d (sin espejos)" % k, lambda k=k: scoreKNN(X_fit, y_fit, X_val, k, 120)))
    for gamma in (0.2, 0.5):
        runs.append(("svm rbf gamma=%.1f/6 C=10 (sin espejos)" % gamma, lambda g=gamma: scoreSVM(X_fit, y_fit, X_val, g / 6)))
    def ridge(X, y_tr):
        W, mu = trainRidge(X, y_tr, 120)
        return scoreRidge(X_val, W, mu)
    runs.append(("ridge (sin espejos)", lambda: ridge(X_fit, y_fit)))
    runs.append(("ridge (con espejos)", lambda: ridge(X_flip, y_flip)))
    done = set()
    if os.path.exists(RESULTADOS + "/comparar.csv"):
        done = set(l.split(",")[0] for l in open(RESULTADOS + "/comparar.csv").read().split("\n") if l)
    for name, run in runs:
        if name in done:
            continue
        if time.time() > t_end:
            print("se acabo el tiempo, vuelve a ejecutar para seguir", flush=True)
            return
        top1, top3 = getAccuracy(run(), y[val])
        saveRow("comparar.csv", [name, "%.4f" % top1, "%.4f" % top3])
        print(name, "top1 %.3f top3 %.3f" % (top1, top3), flush=True)


# ---------- confianza ----------

# pasar las notas a "probabilidades" con softmax: p = exp(nota/t) / suma
def getProba(S:np.ndarray, t:float) -> np.ndarray:
    z = S / t
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


# elegir t para que las probabilidades sean honestas (la que menos se equivoca en validacion)
def fitTemperature(S:np.ndarray, y_true:np.ndarray) -> float:
    best_t, best_loss = 1.0, 1e9
    for t in np.exp(np.linspace(np.log(0.005), np.log(1.0), 60)):
        p = getProba(S, t)
        loss = -np.log(p[np.arange(len(y_true)), y_true] + 1e-9).mean()
        if loss < best_loss:
            best_t, best_loss = t, loss
    return best_t


# umbral de confianza: el valor mas bajo a partir del cual, cuando decimos una sola raza, acertamos el % pedido
# (si no se consigue, devuelve 1.01 y nunca diremos una sola raza)
def fitThreshold(p_top:np.ndarray, correct:np.ndarray, min_accuracy:float) -> float:
    for thr in np.linspace(0.05, 0.99, 95):
        mask = p_top >= thr
        if mask.sum() >= 30 and correct[mask].mean() >= min_accuracy:
            return float(thr)
    return 1.01


# acierto y porcentaje de fotos que quedan como "una sola raza" para varios umbrales
def thresholdTable(p_top:np.ndarray, correct:np.ndarray, top3_ok:np.ndarray) -> list:
    rows = []
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        mask = p_top >= thr
        rows.append([thr, "%.3f" % mask.mean(),
                     "%.3f" % (correct[mask].mean() if mask.any() else 0),
                     "%.3f" % (top3_ok[~mask].mean() if (~mask).any() else 0)])
    return rows


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


# paso final: entrenar con todo train, medir en test (solo una vez), calibrar la confianza y guardar el modelo
def stepFinal(data:dict, names:list, breeds:list, y:np.ndarray, idx_train:np.ndarray, idx_test:np.ndarray, fit:np.ndarray, val:np.ndarray) -> None:
    os.makedirs(RESULTADOS, exist_ok=True)
    open(RESULTADOS + "/final.csv", "w").write("conjunto,fotos,top1,top3\n")    # empezamos el fichero de cero
    # 1. confianza: modelo entrenado solo con fit, y sus notas en val (fotos que no ha visto)
    scaler = fitScaler(data, fit, CLAVES)
    X, y_tr = getTrainSet(data, y, fit, CLAVES, scaler)
    W, mu = trainRidge(X, y_tr, 120)
    S_val = scoreRidge(buildMatrix(data, val, CLAVES, scaler), W, mu)
    t = fitTemperature(S_val, y[val])
    p_val = getProba(S_val, t)
    ok_val = S_val.argmax(axis=1) == y[val]
    thr = fitThreshold(p_val.max(axis=1), ok_val, 0.7)
    print("validacion:", "top1 %.3f top3 %.3f" % getAccuracy(S_val, y[val]), "| t = %.4f" % t, "| umbral = %.2f" % thr, flush=True)
    del X

    # 2. modelo final: todas las fotos de train (con espejos), medido en test
    scaler = fitScaler(data, idx_train, CLAVES)
    X, y_tr = getTrainSet(data, y, idx_train, CLAVES, scaler)
    W, mu = trainRidge(X, y_tr, 120)
    del X
    X_test = buildMatrix(data, idx_test, CLAVES, scaler)
    S = scoreRidge(X_test, W, mu)
    top1, top3 = getAccuracy(S, y[idx_test])
    print("TEST (", len(idx_test), "fotos ): top1 %.4f top3 %.4f" % (top1, top3), flush=True)
    saveRow("final.csv", ["test", len(idx_test), "%.4f" % top1, "%.4f" % top3])

    # 3. acierto por raza y razas que mas se confunden
    order = np.argsort(-S, axis=1)[:, :3]
    f = open(RESULTADOS + "/acierto_por_raza.csv", "w")
    f.write("raza,fotos_test,top1,top3\n")
    for b in range(120):
        m = y[idx_test] == b
        f.write("%s,%d,%.3f,%.3f\n" % (breeds[b], m.sum(), (order[m, 0] == b).mean(), (order[m] == b).any(axis=1).mean()))
    f.close()
    pairs = {}
    for i in range(len(idx_test)):
        if order[i, 0] != y[idx_test][i]:
            pairs[(y[idx_test][i], order[i, 0])] = pairs.get((y[idx_test][i], order[i, 0]), 0) + 1
    f = open(RESULTADOS + "/confusiones.csv", "w")
    f.write("raza_real,raza_dicha,fotos\n")
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:30]:
        f.write("%s,%s,%d\n" % (breeds[a], breeds[b], n))
    f.close()

    # 4. confianza en test con la t y el umbral elegidos en validacion
    p = getProba(S, t)
    p_top = p.max(axis=1)
    correct = order[:, 0] == y[idx_test]
    top3_ok = (order == y[idx_test][:, np.newaxis]).any(axis=1)
    f = open(RESULTADOS + "/umbral.csv", "w")
    f.write("umbral,fotos_con_una_sola_raza,acierto_de_esas,top3_del_resto\n")
    for row in thresholdTable(p_top, correct, top3_ok):
        f.write(",".join(str(v) for v in row) + "\n")
    f.close()
    single = p_top >= thr
    print("con umbral %.2f: una sola raza en %.1f%% de fotos, acierto %.1f%%; en el resto top3 = %.1f%%" %
          (thr, single.mean() * 100, correct[single].mean() * 100 if single.any() else 0, top3_ok[~single].mean() * 100), flush=True)

    # 4b. solo las fotos de stanford (las 3 razas de oxford suben el acierto porque sus fotos tienen otro "estilo")
    source = readSources()
    is_st = np.array([source[names[i]] == "stanford" for i in idx_test])
    t1, t3 = getAccuracy(S[is_st], y[idx_test][is_st])
    saveRow("final.csv", ["test_solo_stanford", is_st.sum(), "%.4f" % t1, "%.4f" % t3])
    print("solo stanford: %d fotos top1 %.4f top3 %.4f" % (is_st.sum(), t1, t3), flush=True)

    # 5. fotos dudosas del test y fotos descartadas (desastres) del test
    info = readList()
    dudosas = set(l[:-4] for l in open(carpeta + "/dudosas.txt").read().split("\n") if l)
    is_dud = np.array([names[i] in dudosas for i in idx_test])
    if is_dud.any():
        t1, t3 = getAccuracy(S[is_dud], y[idx_test][is_dud])
        saveRow("final.csv", ["test_dudosas", is_dud.sum(), "%.4f" % t1, "%.4f" % t3])
        t1b, t3b = getAccuracy(S[~is_dud], y[idx_test][~is_dud])
        saveRow("final.csv", ["test_no_dudosas", (~is_dud).sum(), "%.4f" % t1b, "%.4f" % t3b])
        print("dudosas: %d fotos top1 %.3f top3 %.3f | no dudosas: top1 %.3f top3 %.3f" % (is_dud.sum(), t1, t3, t1b, t3b), flush=True)
    desc_names = [n[:-4] for n in os.listdir(descartadas) if n.endswith(".png")]
    extra, keep = loadExtra(desc_names, "datos/descriptores_descartadas", breeds, info)
    keep_test = [i for i, n in enumerate(keep) if info[n][1] == "test"]
    if keep_test:
        Xd = []
        for key in CLAVES:
            x = extra[key][keep_test]
            if key in RAIZ:
                x = np.sqrt(np.maximum(x, 0))
            mean, std = scaler[key]
            Xd.append(((x - mean) / std / np.sqrt(x.shape[1])).astype(np.float32))
        yd = np.array([breeds.index(info[keep[i]][0]) for i in keep_test])
        t1, t3 = getAccuracy(scoreRidge(np.hstack(Xd), W, mu), yd)
        saveRow("final.csv", ["test_descartadas", len(yd), "%.4f" % t1, "%.4f" % t3])
        print("descartadas (desastres) en test: %d fotos top1 %.3f top3 %.3f" % (len(yd), t1, t3), flush=True)

    # 6. guardar el modelo para reconocer_raza.py
    os.makedirs(modelo, exist_ok=True)
    pickle.dump({"breeds": breeds, "keys": CLAVES, "scaler": scaler, "W": W, "mu": mu, "t": t, "thr": thr}, open(modelo + "/modelo.pkl", "wb"))
    print("modelo guardado", flush=True)


# paso razas: como cambia el acierto cuando hay mas razas (entrenamos solo con las razas elegidas)
# se hace con 3 ordenes de razas al azar y se saca la media (nos da una idea del ruido)
def stepBreeds(y:np.ndarray, idx_train:np.ndarray, idx_test:np.ndarray, t_end:float) -> None:
    data = loadData()
    scaler = fitScaler(data, idx_train, CLAVES)
    X_all = buildMatrix(data, idx_train, CLAVES, scaler)
    X_all_flip = buildMatrix(data, idx_train, CLAVES, scaler, True)
    X_test = buildMatrix(data, idx_test, CLAVES, scaler)
    del data    # liberamos memoria
    done = set()
    if os.path.exists(RESULTADOS + "/acierto_vs_razas.csv"):
        done = set(tuple(l.split(",")[:2]) for l in open(RESULTADOS + "/acierto_vs_razas.csv").read().split("\n") if l)
    for n in (5, 10, 20, 40, 60, 80, 100):
        for seed in range(3):
            if (str(n), str(seed)) in done:
                continue
            if time.time() > t_end:
                print("se acabo el tiempo, vuelve a ejecutar para seguir", flush=True)
                return
            chosen = np.random.RandomState(seed).permutation(120)[:n]
            new_label = -np.ones(120, int)
            new_label[chosen] = np.arange(n)
            m_tr = new_label[y[idx_train]] >= 0
            m_te = new_label[y[idx_test]] >= 0
            X = np.vstack([X_all[m_tr], X_all_flip[m_tr]])
            y_tr = np.concatenate([new_label[y[idx_train][m_tr]]] * 2)
            W, mu = trainRidge(X, y_tr, n)
            top1, top3 = getAccuracy(scoreRidge(X_test[m_te], W, mu), new_label[y[idx_test][m_te]])
            saveRow("acierto_vs_razas.csv", [n, seed, "%.4f" % top1, "%.4f" % top3])
            print("razas", n, "orden", seed, "top1 %.3f top3 %.3f" % (top1, top3), flush=True)


# grafica del acierto segun el numero de razas (la de 120 sale del paso final)
def plotBreeds() -> None:
    rows = [l.split(",") for l in open(RESULTADOS + "/acierto_vs_razas.csv").read().split("\n") if l]
    final = [l.split(",") for l in open(RESULTADOS + "/final.csv").read().split("\n")[1:] if l][0]
    ns = sorted(set(int(r[0]) for r in rows)) + [120]
    top1 = [np.mean([float(r[2]) for r in rows if int(r[0]) == n]) * 100 for n in ns[:-1]] + [float(final[2]) * 100]
    top3 = [np.mean([float(r[3]) for r in rows if int(r[0]) == n]) * 100 for n in ns[:-1]] + [float(final[3]) * 100]
    plt.figure(figsize=(7, 4))
    plt.plot(ns, top3, "o-", label="top-3")
    plt.plot(ns, top1, "o-", label="top-1")
    plt.xlabel("numero de razas")
    plt.ylabel("acierto en test (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTADOS + "/acierto_vs_razas.png", dpi=120)
    for n, a, b in zip(ns, top1, top3):
        print(n, "razas: top1 %.1f top3 %.1f" % (a, b))


def main():
    t_end = time.time() + (float(sys.argv[2]) if len(sys.argv) > 2 else 150)
    paso = sys.argv[1] if len(sys.argv) > 1 else "cargar"
    if paso == "cargar":
        stepLoad(t_end)
        return
    y, split = np.load(matrices + "/y.npy"), np.load(matrices + "/split.npy")
    idx_train, idx_test = np.where(split == "train")[0], np.where(split == "test")[0]
    fit, val = splitValidation(y, idx_train)
    if paso == "razas":
        stepBreeds(y, idx_train, idx_test, t_end)
        if time.time() < t_end:
            plotBreeds()
        return
    data = loadData()
    if paso == "descriptores":
        stepDescriptors(data, y, fit, val, t_end)
    if paso == "comparar":
        stepCompare(data, y, fit, val, t_end)
    if paso == "final":
        names, breeds, y_l, split_l = listDataset()
        stepFinal(data, names, breeds, y, idx_train, idx_test, fit, val)


if __name__ == "__main__":
    main()
