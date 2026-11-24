import sys
import os
import streamlit as st
import cv2 as cv
import numpy as np

# trabajar siempre desde la carpeta de este archivo (asi encuentra la carpeta modelo/)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import reconocer_raza as R

# ------------------------------------------------------------
# aplicacion web para reconocer la raza de un perro (version 2: 120 razas)
#
# uso:  streamlit run app.py
#       (se abre sola en el navegador, en http://localhost:8501)
#
# pestana 1: subes una foto y salen las 3 razas que mas se parecen
# pestana 2: especificaciones (como funciona, datos y resultados)
# ------------------------------------------------------------

st.set_page_config(page_title="Reconocedor de razas de perro", page_icon="🐶")


# el modelo se carga una sola vez (si no, tardaria en cada foto)
@st.cache_resource
def getModel() -> dict:
    return R.loadModel(R.modelo)


# leer la foto que sube la persona (bytes -> imagen de opencv)
def readUpload(file) -> np.ndarray:
    data = np.frombuffer(file.getvalue(), np.uint8)
    return cv.imdecode(data, cv.IMREAD_COLOR)


def showRecognize() -> None:
    st.write("Sube la foto de un perro y te digo las 3 razas a las que más se parece.")
    st.caption("Mejor si hay un solo perro, de frente, con la cara bien visible y sin mucha gente alrededor.")
    file = st.file_uploader("Foto del perro", type=["jpg", "jpeg", "png", "webp"])
    if file is None:
        return
    img = readUpload(file)
    if img is None:
        st.error("No he podido leer esa foto, prueba con otra (jpg o png).")
        return
    with st.spinner("Mirando al perro..."):
        proba, result, doubt = R.predictBreed(img, getModel())
    col1, col2 = st.columns(2)
    col1.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), caption="tu foto")
    col2.image(cv.cvtColor(result, cv.COLOR_BGR2RGB), caption="el perro recortado")
    st.subheader("Las 3 razas que más se parecen")
    for n, i in enumerate(np.argsort(-proba)[:3]):
        st.markdown("**" + str(n + 1) + ".** " + R.prettyName(getModel()["breeds"][i]).title())
    if doubt:
        st.warning("El recorte del perro no es muy fiable. Prueba con otra foto: un solo perro, de frente y con poco fondo.")
    st.caption("Solo conozco 120 razas. Si tu perro es mestizo o de otra raza, te enseñaré las que más se parecen.")


def showSpecs() -> None:
    st.markdown("""
### Cómo funciona
Está hecho solo con técnicas clásicas de visión por computador, **sin deep learning**.
1. **Recortar el perro:** saliency + GrabCut. Se quita el fondo (queda negro) y se deja en 256×256.
2. **Describir la foto** (solo dentro del perro): color (HSV), HOG, LBP (textura), Bag of Words con SIFT denso y Fisher vector.
3. **Clasificar:** regresión ridge con todas esas características. La raza con más nota va la primera.

### Datos
- Dataset *Stanford Dogs* (117 razas) y 3 razas de *Oxford-IIIT Pet* (shiba inu, havanese y american bulldog): **120 razas**.
- 20.234 fotos después de quitar 452 recortes que salieron muy mal.
- Entrenamiento con la división oficial: 100 fotos por raza.

### Resultados (8.483 fotos de test que el modelo no vio)
- La raza correcta sale la **primera** en el 31% de las fotos.
- Está **entre las 3 primeras** en el 49%.
- Con menos razas acierta más: con 20 razas, 54% y 74%.

### Límites
- Solo conoce esas 120 razas. Con un mestizo o una raza que no sabe, dice las que más se parecen.
- Funciona mejor con un solo perro, de frente y con la cara visible.
""")


st.title("🐶 ¿De qué raza es tu perro?")
tab1, tab2 = st.tabs(["Reconocer", "Especificaciones"])
with tab1:
    showRecognize()
with tab2:
    showSpecs()
