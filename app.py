import sys
import os
import streamlit as st
import cv2 as cv
import numpy as np

# buscar el codigo del reconocedor: en la carpeta de al lado (codigo/) o en esta misma (cuando esta subida a internet)
BASE = os.path.dirname(os.path.abspath(__file__))
for folder in (BASE, os.path.join(BASE, "..", "codigo")):
    if os.path.exists(os.path.join(folder, "reconocer_raza.py")):
        sys.path.insert(0, folder)
        break
os.chdir(BASE)    # asi encuentra las carpetas modelo/ y ejemplos/ tambien en internet
import reconocer_raza as R
import preparar_dataset as P
import descriptores as D

# ------------------------------------------------------------
# aplicacion web para reconocer la raza de un perro (version 2: 120 razas)
#
# uso:  streamlit run app_web/app_web.py     (se abre sola en el navegador)
#
# pestana 1 (reconocer): recomendaciones, subir la foto (o elegir un ejemplo) y las 3 razas
# pestana 2 (especificaciones): lo que hace el codigo con la foto que se ha subido
# ------------------------------------------------------------

EJEMPLOS = "ejemplos"    # fotos de ejemplo (nombre del fichero = raza)

st.set_page_config(page_title="Reconocedor de razas de perro", page_icon="🐕", layout="centered")

# tamanos de letra: texto 1.2rem, titulos 1.7rem, pestanas 1.4rem
CSS = """
<style>
.stApp, .stApp p, .stApp li, .stApp td, .stApp th {font-size: 1.2rem;}
.stApp h3 {font-size: 1.7rem !important;}
[data-testid="stHeaderActionElements"], .stApp h1 a, .stApp h2 a, .stApp h3 a {display: none !important;}
[data-testid="stAlert"] p, [data-testid="stAlert"] div {font-size: 1.2rem !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {font-size: 1.1rem !important;}
[data-testid="stFileUploader"] section, [data-testid="stFileUploader"] section * {font-size: 1.2rem !important;}
.stTabs button, .stTabs button *, button[role="tab"], button[role="tab"] * {font-size: 1.4rem !important; font-weight: 600 !important;}
.stTabs button, button[role="tab"] {padding: 0.9rem 1.6rem !important; height: auto !important;}

.stButton > button {width: 100%; min-height: 3.4rem; padding: 0.4rem 0.6rem;}
.stButton > button p {font-size: 1.05rem !important; line-height: 1.2; white-space: normal !important; text-align: center;}

.hero {background: linear-gradient(135deg, #1f4d3a 0%, #3f8160 100%); border-radius: 12px; padding: 2.2rem 2rem 1.8rem 2rem; margin-bottom: 1.4rem; text-align: center;}
.hero h1 {color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0; padding: 0; line-height: 1.25;}
.hero p {color: #e3f1e9; font-size: 1.25rem; margin: 0.6rem 0 0 0;}
.tips {border: 1px solid #cfe3d6; border-left: 5px solid #3f8160; background: #f1f8f4; color: #22332a; border-radius: 8px; padding: 1.1rem 1.5rem; margin: 0.8rem 0 1rem 0; line-height: 1.55;}
.tips h3 {color: #1f4d3a; margin: 0 0 0.4rem 0; padding: 0;}
.tips ul {margin: 0.3rem 0 0.4rem 0; padding-left: 1.2rem;}
.tips li {margin: 0.2rem 0; color: #22332a;}
.title {color: #1f4d3a; font-size: 1.7rem; font-weight: 700; line-height: 1.25; margin: 1.4rem 0 0.5rem 0;}
.subtitle {color: #4b5d53; margin: 0 0 1rem 0;}
.card {display: flex; align-items: center; gap: 1.1rem; background: #ffffff; color: #1f2937; border-radius: 8px; padding: 1.1rem 1.5rem; margin: 0.7rem 0; border: 1px solid #d5e5db;}
.card .num {font-size: 2.6rem; line-height: 1; min-width: 3rem; text-align: center;}
.card .name {font-size: 1.5rem; font-weight: 600; color: #1f2937;}
.note {display: flex; gap: 0.8rem; align-items: flex-start; background: #f6f8f7; border: 1px solid #e0e7e3; border-radius: 8px; padding: 0.9rem 1.2rem; margin: 1.2rem 0 0.4rem 0; color: #4b5d53; line-height: 1.55;}
.note .ico {font-size: 1.6rem; line-height: 1.3;}
.step {background: #f1f8f4; color: #22332a; border-left: 5px solid #3f8160; border-radius: 4px; padding: 0.7rem 1.1rem; margin: 1.6rem 0 0.6rem 0;}
.step b {color: #1f4d3a;}
.data {background: #f6f8f7; border-left: 3px solid #3f8160; border-radius: 4px; padding: 0.5rem 1rem; margin: 0.6rem 0 0.8rem 0; color: #22332a;}
</style>
"""


# el modelo se carga una sola vez (si no, tardaria en cada foto)
@st.cache_resource
def getModel() -> dict:
    return R.loadModel(R.modelo)


# opencv trabaja en BGR y la web en RGB
def toRGB(img:np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


# nombre de una raza para mostrarlo
def breedName(i:int) -> str:
    return R.prettyName(getModel()["breeds"][i]).title()


# las 3 razas con mas nota
def getTop3(scores:np.ndarray) -> list:
    return [breedName(i) for i in np.argsort(-scores)[:3]]


# lo que hace el codigo con la foto, para ensenarlo en especificaciones
def getSteps(img:np.ndarray, result:np.ndarray, result_mask:np.ndarray) -> dict:
    steps = {}
    cv.setRNGSeed(0)
    small = P.resizeMax(img, P.WORK)
    steps["small"] = small
    # saliency: donde mira la atencion
    sal_map = P.getSaliencyMap(small)
    steps["saliency"] = cv.applyColorMap(sal_map, cv.COLORMAP_JET)
    # otsu + contornos: rectangulo del perro
    otsu, thresh = cv.threshold(sal_map, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    steps["otsu"] = otsu
    steps["thresh"] = cv.morphologyEx(cv.threshold(sal_map, 0.4*otsu, 255, cv.THRESH_BINARY)[1], cv.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
    rect = P.getRect(sal_map, 0.4*otsu, 0.15)
    steps["rect_size"] = (rect[2], rect[3])
    boxed = small.copy()
    cv.rectangle(boxed, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 3)
    steps["rect"] = boxed
    # grabcut: separa el perro del fondo pixel a pixel
    mask = P.applyGrabCut(small, rect)
    steps["grabcut"] = small * mask[:, :, np.newaxis]
    # descriptores sobre el recorte final
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    lbp = D.getLBPImage(gray, 1)
    lbp_full = np.zeros(gray.shape, np.uint8)
    lbp_full[1:-1, 1:-1] = lbp
    steps["lbp"] = lbp_full * (result_mask > 0)
    points = result.copy()
    n_points = 0
    for y in range(4, D.SIZE, 8):
        for x in range(4, D.SIZE, 8):
            if result_mask[y, x] > 0:
                cv.circle(points, (x, y), 2, (0, 255, 255), -1)
                n_points += 1
    steps["sift"] = points
    steps["n_points"] = n_points
    steps["n_patches"] = len(D.getLocalColor(result, result_mask))
    steps["dog_percent"] = 100 * cv.countNonZero(result_mask) / result_mask.size
    hsv = cv.cvtColor(result, cv.COLOR_BGR2HSV)
    steps["hue"] = cv.calcHist([hsv], [0], result_mask, [18], [0, 180]).flatten()
    return steps


# analizar una foto: razas, recorte y todo lo de especificaciones
def analyze(data:bytes) -> dict:
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        return None
    scores, result, result_mask, doubt = R.predictBreed(img, getModel())
    steps = getSteps(img, result, result_mask)
    return {"img": img, "scores": scores, "result": result, "doubt": doubt, "steps": steps}


# ---------- pestana 1: reconocer ----------

# al subir una foto o elegir un ejemplo apuntamos cual se usa
def useUpload() -> None:
    st.session_state["source"] = ("subida", "")


def useExample(name:str) -> None:
    st.session_state["source"] = ("ejemplo", name)


# miniatura cuadrada de una foto de ejemplo (todas del mismo tamano para que queden alineadas)
def getThumbnail(path:str) -> np.ndarray:
    img = cv.imread(path)
    h, w = img.shape[:2]
    side = min(h, w)
    y, x = (h - side) // 2, (w - side) // 2
    square = img[y:y + side, x:x + side]
    return cv.cvtColor(cv.resize(square, (300, 300), interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)


# fotos de ejemplo (para probar sin tener una foto a mano)
def showExamples() -> None:
    names = sorted(os.listdir(EJEMPLOS)) if os.path.isdir(EJEMPLOS) else []
    if len(names) == 0:
        return
    st.markdown('<div class="title">O prueba con un ejemplo</div>', unsafe_allow_html=True)
    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        col.image(getThumbnail(EJEMPLOS + "/" + name))
        col.button(R.prettyName(name[:-4]).title(), key="ejemplo_" + name, on_click=useExample, args=(name,))


# bytes de la foto que se esta usando (None si todavia no hay)
def getPhoto(file) -> tuple:
    source = st.session_state.get("source")
    if source is not None and source[0] == "ejemplo":
        return source, open(EJEMPLOS + "/" + source[1], "rb").read()
    if file is not None:
        return ("subida", file.name, file.size), file.getvalue()
    return None, None


def showRecognize() -> None:
    st.markdown("""
<div class="tips">
<h3>Recomendaciones para la foto</h3>
Para un mejor resultado:
<ul>
<li>Un solo perro, de frente y con la cara visible.</li>
<li>Cuerpo entero (o casi) y sin personas alrededor.</li>
<li>Imagen nítida y con buena luz.</li>
</ul>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="title">📷 Sube la foto de tu perro</div>', unsafe_allow_html=True)
    file = st.file_uploader("Sube la foto de tu perro", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed", on_change=useUpload)
    showExamples()
    key, data = getPhoto(file)
    if data is None:
        return
    if st.session_state.get("key") != key:
        with st.spinner("Mirando al perro... 🔍"):
            a = analyze(data)
        if a is None:
            st.error("No he podido leer esa foto. Prueba con otra (jpg o png).")
            return
        st.session_state["key"] = key
        st.session_state["analisis"] = a
    a = st.session_state["analisis"]
    st.image(toRGB(a["img"]), caption="Tu foto", width=380)
    st.markdown('<div class="title">🐾 Tu perro se parece a estas 3 razas</div>'
                '<div class="subtitle">Las razas más parecidas, sin ningún orden concreto.</div>', unsafe_allow_html=True)
    # por orden alfabetico, para que no parezca que hay una primera y una ultima
    html = ""
    for name in sorted(getTop3(a["scores"])):
        html += '<div class="card"><div class="num">🐕</div><div class="name">' + name + '</div></div>'
    st.markdown(html, unsafe_allow_html=True)
    if a["doubt"]:
        st.warning("Este resultado es poco fiable.")
    st.markdown('<div class="note"><div class="ico">💡</div><div>Si tu perro es mestizo o de una raza que no está entre las 120 que '
                'reconoce, verás las más parecidas. En la pestaña <b>Especificaciones</b> puedes ver cómo se ha analizado tu foto.</div></div>', unsafe_allow_html=True)


# ---------- pestana 2: especificaciones ----------

# titulo de cada apartado
def stepTitle(text:str) -> None:
    st.markdown('<div class="step">' + text + '</div>', unsafe_allow_html=True)


# dato concreto de la foto que se ha subido
def showData(text:str) -> None:
    st.markdown('<div class="data">' + text + '</div>', unsafe_allow_html=True)


def showSpecs() -> None:
    st.markdown("### Cómo funciona")
    st.write("El programa localiza al perro, lo recorta, lo describe con números y decide la raza. "
             "Todo con visión por computador clásica, sin deep learning.")
    a = st.session_state.get("analisis")
    if a is None:
        st.info("Sube una foto en la pestaña **Reconocer** y aquí verás lo que ha hecho el programa con ella.")
    s = a["steps"] if a is not None else None

    stepTitle("<b>Localización del perro</b> (saliency + Otsu + contornos)")
    st.write("*Saliency* marca lo que más llama la atención de la foto. Con **Otsu** y contornos se saca un rectángulo alrededor del perro.")
    if s is not None:
        c = st.columns(4)
        c[0].image(toRGB(s["small"]), caption="foto reducida")
        c[1].image(toRGB(s["saliency"]), caption="mapa de saliency")
        c[2].image(toRGB(s["thresh"]), caption="umbral de Otsu")
        c[3].image(toRGB(s["rect"]), caption="rectángulo del perro")
        h, w = s["small"].shape[:2]
        showData("Foto reducida a %d×%d · umbral de Otsu: %d · rectángulo: %d×%d píxeles" % (w, h, s["otsu"], s["rect_size"][0], s["rect_size"][1]))

    stepTitle("<b>Segmentación del perro</b> (GrabCut)")
    st.write("*GrabCut* separa el perro del fondo dentro del rectángulo. El fondo se pone negro y el recorte queda en 256×256.")
    if s is not None:
        c = st.columns(3)
        c[0].image(toRGB(s["grabcut"]), caption="perro separado del fondo")
        c[1].image(toRGB(a["result"]), caption="recorte final 256×256")
        c[2].write("")
        if a["doubt"]:
            showData("Recorte dudoso: casi no hay perro (o casi todo es perro) dentro del rectángulo. Por eso el resultado se marca como poco fiable.")
        else:
            showData("Recorte normal: el perro ocupa una parte razonable del rectángulo.")

    stepTitle("<b>Descripción de la imagen</b> (7.876 números por foto)")
    st.write("Solo se mira dentro del perro, no el fondo negro. Cada descriptor mide algo distinto:")
    st.markdown("""
| Descriptor | Qué mide | Números |
| --- | --- | --- |
| Color HSV | los colores del pelo | 152 |
| HOG | la dirección de los bordes | 324 |
| LBP (2 tamaños) | la textura del pelo | 1.536 |
| Bag of Words | puntos de interés SIFT agrupados con k-means | 1.000 |
| Fisher vector | SIFT + PCA + mezcla de gaussianas | 4.096 |
| Color local | color (Lab) de parches de 16×16 + mezcla de gaussianas | 768 |
""")
    if s is not None:
        c = st.columns(3)
        c[0].image(toRGB(s["sift"]), caption="puntos donde se mira (SIFT denso)")
        c[1].image(toRGB(s["lbp"]), caption="textura (LBP)")
        c[2].caption("colores del perro (tono)")
        c[2].bar_chart(s["hue"])
        showData("El perro ocupa el %d %% del recorte · SIFT mira %d puntos (en 3 tamaños) · el color local usa %d parches" %
                 (round(s["dog_percent"]), s["n_points"], s["n_patches"]))

    stepTitle("<b>Clasificación de la raza</b> (LDA)")
    st.write("El clasificador da una nota a cada una de las 120 razas y se muestran las 3 con más nota.")
    if a is not None:
        st.success("Para tu foto: " + " · ".join(getTop3(a["scores"])))


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>🐶 Reconocedor de razas de perro 🐾</h1><p>Sube una foto de tu perro 🦴 y descubre a qué razas se parece</p></div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🐕 Reconocer", "🔬 Especificaciones"])
    with tab1:
        showRecognize()
    with tab2:
        showSpecs()


main()
