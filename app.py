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
os.chdir(BASE)    # asi encuentra la carpeta modelo/ tambien en internet
import reconocer_raza as R
import preparar_dataset as P
import descriptores as D

# ------------------------------------------------------------
# aplicacion web para reconocer la raza de un perro (version 2: 120 razas)
#
# uso:  streamlit run app_web/app_web.py     (se abre sola en el navegador)
#
# pestana 1 (reconocer): condiciones de la foto, subir la foto y las 3 razas
# pestana 2 (especificaciones): todo lo que hace el codigo con la foto, paso a paso
# ------------------------------------------------------------

st.set_page_config(page_title="Reconocedor de razas de perro", page_icon="🐕", layout="centered")

CSS = """
<style>
.hero {position: relative; background: linear-gradient(135deg, #1f4d3a 0%, #3f8160 100%); border-radius: 12px; padding: 2.2rem 2rem 1.8rem 2rem; color: #ffffff; margin-bottom: 1.4rem; text-align: center;}
.hero h1 {color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0; padding: 0; line-height: 1.25;}
.hero p {color: #e3f1e9; font-size: 1.02rem; margin: 0.6rem 0 0 0;}
.tips {border: 1px solid #cfe3d6; border-left: 5px solid #3f8160; background: #f1f8f4; color: #22332a; border-radius: 8px; padding: 1.1rem 1.5rem; margin: 0.8rem 0 1rem 0; font-size: 1.15rem; line-height: 1.55;}
.tips h3 {color: #1f4d3a; margin: 0 0 0.4rem 0; padding: 0; font-size: 1.3rem;}
.tips ul {margin: 0.3rem 0 0.4rem 0; padding-left: 1.2rem;}
.tips li {margin: 0.2rem 0; color: #22332a;}
.result-head {margin: 1.6rem 0 1rem 0;}
.result-head h2 {color: #1f4d3a; font-size: 2.2rem; font-weight: 700; line-height: 1.25; margin: 0; padding: 0;}
.result-head p {color: #4b5d53; font-size: 1.25rem; margin: 0.4rem 0 0 0;}
.card {display: flex; align-items: center; gap: 1.1rem; background: #ffffff; color: #1f2937; border-radius: 8px;
       padding: 1.1rem 1.5rem; margin: 0.7rem 0; border: 1px solid #d5e5db;}
.card .num {font-size: 2.6rem; line-height: 1; min-width: 3rem; text-align: center;}
.card .name {font-size: 1.6rem; font-weight: 600; color: #1f2937;}
.note {display: flex; gap: 0.8rem; align-items: flex-start; background: #f6f8f7; border: 1px solid #e0e7e3; border-radius: 8px; padding: 0.9rem 1.2rem; margin: 1.2rem 0 0.4rem 0; color: #4b5d53; font-size: 1.15rem; line-height: 1.55;}
.note .ico {font-size: 1.6rem; line-height: 1.3;}
.step {background: #f1f8f4; color: #22332a; border-left: 5px solid #3f8160; border-radius: 4px; padding: 0.7rem 1.1rem; margin: 1.4rem 0 0.6rem 0;}
.step b {color: #1f4d3a;}
.small {color: #6b7280; font-size: 1rem;}
[data-testid="stFileUploader"] label p {font-size: 1.3rem !important; font-weight: 600;}
[data-testid="stAlert"] p {font-size: 1.2rem !important;}
[data-testid="stCaptionContainer"] {font-size: 1.05rem !important;}
.stMarkdown p, .stMarkdown li {font-size: 1.12rem;}
button[data-baseweb="tab"] p {font-size: 1.15rem !important;}
</style>
"""


# el modelo se carga una sola vez (si no, tardaria en cada foto)
@st.cache_resource
def getModel() -> dict:
    return R.loadModel(R.modelo)


# leer la foto que sube la persona (bytes -> imagen de opencv)
def readUpload(file) -> np.ndarray:
    data = np.frombuffer(file.getvalue(), np.uint8)
    return cv.imdecode(data, cv.IMREAD_COLOR)


# opencv trabaja en BGR y la web en RGB
def toRGB(img:np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


# todos los pasos que hace el codigo con la foto (para ensenarlos en especificaciones)
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
    steps["thresh"] = cv.morphologyEx(cv.threshold(sal_map, 0.4*otsu, 255, cv.THRESH_BINARY)[1], cv.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
    rect = P.getRect(sal_map, 0.4*otsu, 0.15)
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
    for y in range(4, D.SIZE, 8):
        for x in range(4, D.SIZE, 8):
            if result_mask[y, x] > 0:
                cv.circle(points, (x, y), 2, (0, 255, 255), -1)
    steps["sift"] = points
    hsv = cv.cvtColor(result, cv.COLOR_BGR2HSV)
    steps["hue"] = cv.calcHist([hsv], [0], result_mask, [18], [0, 180]).flatten()
    return steps


# pestana 1: condiciones, subir la foto y las 3 razas
def showRecognize() -> None:
    st.markdown("""
<div class="tips">
<h3>📸 La foto ideal</h3>
Si puedes elegir, una foto así suele funcionar genial:
<ul>
<li>Un solo perro, de frente y con la cara bien a la vista.</li>
<li>Con el cuerpo entero (o casi) y sin personas alrededor.</li>
<li>Nítida y con buena luz.</li>
</ul>
Si no la tienes así, también puedes probar 😉
</div>
""", unsafe_allow_html=True)
    file = st.file_uploader("📷 Sube la foto de tu perro", type=["jpg", "jpeg", "png", "webp"])
    if file is None:
        return
    key = (file.name, file.size)
    if st.session_state.get("key") != key:
        img = readUpload(file)
        if img is None:
            st.error("No he podido leer esa foto. Prueba con otra (jpg o png).")
            return
        with st.spinner("Mirando al perro... 🔍"):
            proba, result, doubt = R.predictBreed(img, getModel())
            ret, result_mask, doubt2, box = P.processImage(img)
            steps = getSteps(img, result, result_mask)
        st.session_state["key"] = key
        st.session_state["analisis"] = {"img": img, "proba": proba, "result": result, "doubt": doubt, "steps": steps}
    a = st.session_state["analisis"]
    st.image(toRGB(a["img"]), caption="Tu foto", width=380)
    st.markdown('<div class="result-head"><h2>🐾 Tu perro se parece a estas 3 razas</h2>'
                '<p>Las razas más parecidas, sin ningún orden concreto.</p></div>', unsafe_allow_html=True)
    # las ponemos por orden alfabetico para que no parezca que hay una primera y una ultima
    names = sorted(R.prettyName(getModel()["breeds"][i]).title() for i in np.argsort(-a["proba"])[:3])
    html = ""
    for name in names:
        html += '<div class="card"><div class="num">🐕</div><div class="name">' + name + '</div></div>'
    st.markdown(html, unsafe_allow_html=True)
    if a["doubt"]:
        st.warning("Este resultado es poco fiable.")
    st.markdown('<div class="note"><div class="ico">💡</div><div>Si tu perro es mestizo o de una raza que no está entre las 120 que '
                'reconoce, verás las más parecidas. En la pestaña <b>Especificaciones</b> puedes ver cómo se ha analizado tu foto.</div></div>', unsafe_allow_html=True)


# titulo de un paso
def stepTitle(text:str) -> None:
    st.markdown('<div class="step">' + text + '</div>', unsafe_allow_html=True)


# pestana 2: lo que hace el codigo con la foto
def showSpecs() -> None:
    st.markdown("### 🐾 Cómo funciona")
    st.write("Todo está hecho con técnicas clásicas de visión por computador, **sin deep learning**. "
             "Aquí se ve el procesamiento completo de tu foto: localización, segmentación, descripción y clasificación.")
    a = st.session_state.get("analisis")
    if a is None:
        st.info("Sube una foto en la pestaña **Reconocer** y aquí verás, con tu foto, todo lo que hace el programa.")
    s = a["steps"] if a is not None else None

    stepTitle("<b>Localización del perro</b> (saliency + Otsu + contornos)")
    st.write("El mapa de *saliency* marca lo que más llama la atención (en rojo). Con el umbral de **Otsu** y "
             "`findContours` + `boundingRect` se obtiene un rectángulo alrededor del perro.")
    if s is not None:
        c = st.columns(4)
        c[0].image(toRGB(s["small"]), caption="foto reducida")
        c[1].image(toRGB(s["saliency"]), caption="mapa de saliency")
        c[2].image(toRGB(s["thresh"]), caption="umbral de Otsu")
        c[3].image(toRGB(s["rect"]), caption="rectángulo del perro")

    stepTitle("<b>Segmentación del perro</b> (GrabCut)")
    st.write("*GrabCut* recibe el rectángulo y decide, píxel a píxel, qué es perro y qué es fondo. "
             "El fondo se pone negro y el perro se recorta y se deja en **256×256** sin deformarlo. "
             "Si el resultado sale raro, se repite con otro rectángulo.")
    if s is not None:
        c = st.columns(3)
        c[0].image(toRGB(s["grabcut"]), caption="perro separado del fondo")
        c[1].image(toRGB(a["result"]), caption="recorte final 256×256")
        c[2].write("")

    stepTitle("<b>Descripción de la imagen</b> (7.876 números por foto)")
    st.write("Solo se mira dentro del perro (no el fondo negro). Cada descriptor mide una cosa distinta:")
    st.markdown("""
| Descriptor | Qué mide | Números |
| --- | --- | --- |
| Color HSV | los colores del pelo | 152 |
| HOG | la dirección de los bordes | 324 |
| LBP (2 tamaños) | la textura del pelo | 1.536 |
| Bag of Words | puntos de interés SIFT agrupados con k-means | 1.000 |
| Fisher vector | SIFT + PCA + mezcla de gaussianas | 4.096 |
| Color local | color (Lab) de parches de 16×16 dentro del perro + mezcla de gaussianas | 768 |
""")
    if s is not None:
        c = st.columns(3)
        c[0].image(toRGB(s["sift"]), caption="puntos donde se mira (SIFT denso)")
        c[1].image(toRGB(s["lbp"]), caption="textura (LBP)")
        c[2].caption("colores del perro (tono)")
        c[2].bar_chart(s["hue"])

    stepTitle("<b>Clasificación de la raza</b> (LDA, un clasificador lineal)")
    st.write("El clasificador aprende, con 11.751 fotos, cómo es la foto \"media\" de cada una de las 120 razas y cómo se mueven las fotos alrededor de ella. "
             "Luego da una nota a cada raza según lo cerca que esté tu foto de esa media. Se muestran las 3 razas con más nota. "
             "Se entrenó con la división oficial de *Stanford Dogs* (100 fotos por raza) y también con las fotos en espejo.")
    if a is not None:
        st.success("Para tu foto: " + " · ".join(R.prettyName(getModel()["breeds"][i]).title() for i in np.argsort(-a["proba"])[:3]))

    st.markdown("### 📊 Datos y resultados")
    st.markdown("""
- **Dataset:** *Stanford Dogs* (117 razas) y 3 razas de *Oxford-IIIT Pet* (shiba inu, havanese y american bulldog): **120 razas**.
- **Limpieza:** de 20.686 fotos se quitaron 452 donde el recorte falló mucho. Quedan **20.234**.
- **Resultado en test** (8.483 fotos que el modelo no vio): la raza correcta sale la **primera en el 37%** y **entre las 3 primeras en el 55%** (al azar sería 0,8% y 2,5%).
- **Con menos razas acierta más:** con 20 razas, 58% y 79%.
- **Mejoras que más aportaron:** cambiar a LDA (+6 puntos en el top 3), el Fisher vector (+6), las fotos en espejo (+4) y el color local (+4).

### ⚠️ Límites
- Solo conoce esas 120 razas. Con un mestizo o una raza que no sabe, enseña las que más se parecen.
- Funciona mejor con un solo perro, de frente y con la cara visible.
- En algunas razas acierta mucho (american bulldog 81%) y en otras muy poco (american staffordshire terrier 11%).
""")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>🐶 Reconocedor de razas de perro 🐾</h1><p>Sube una foto de tu perro 🦴 y descubre a qué razas se parece</p></div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🐕 Reconocer", "🔬 Especificaciones"])
    with tab1:
        showRecognize()
    with tab2:
        showSpecs()


if __name__ == "__main__":
    main()
