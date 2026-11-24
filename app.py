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

st.set_page_config(page_title="Reconocedor de razas de perro", page_icon="🐶", layout="centered")

CSS = """
<style>
.hero {background: linear-gradient(135deg, #5b50ff 0%, #ff6fb1 100%); border-radius: 22px; padding: 2.2rem 1.5rem;
       text-align: center; color: white; margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(91,80,255,0.25);}
.hero h1 {color: white; font-size: 2.4rem; margin: 0; padding: 0;}
.hero p {color: #fff; font-size: 1.1rem; margin: 0.4rem 0 0 0; opacity: 0.95;}
.must {border: 3px solid #e53935; background: #fff4f4; color: #5c1010; border-radius: 18px; padding: 1.1rem 1.5rem; margin: 0.8rem 0;}
.must h3 {color: #c62828; margin: 0 0 0.5rem 0; padding: 0; font-size: 1.25rem;}
.must ul {margin: 0.3rem 0 0.6rem 0; padding-left: 1.3rem;}
.must li {margin: 0.25rem 0; color: #5c1010;}
.must .warn {font-weight: 700; color: #c62828;}
.card {display: flex; align-items: center; gap: 1rem; background: #ffffff; color: #222; border-radius: 18px;
       padding: 1rem 1.4rem; margin: 0.7rem 0; box-shadow: 0 4px 14px rgba(0,0,0,0.12); border: 1px solid #eee;}
.card .medal {font-size: 2.6rem; line-height: 1;}
.card .name {font-size: 1.6rem; font-weight: 800; color: #2d2a6e;}
.card .tag {font-size: 0.9rem; color: #777;}
.card.first {background: linear-gradient(135deg, #fff8e1 0%, #ffe9a8 100%); border: 2px solid #ffc107;}
.card.first .name {font-size: 2rem;}
.step {background: #f4f3ff; color: #2b2b44; border-left: 6px solid #5b50ff; border-radius: 12px; padding: 0.8rem 1.1rem; margin: 1rem 0 0.5rem 0;}
.step b {color: #3b32c9;}
.step .nuevo {background: #ff6fb1; color: white; border-radius: 8px; padding: 0 0.5rem; font-size: 0.8rem; margin-left: 0.4rem;}
.small {color: #777; font-size: 0.9rem;}
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
<div class="must">
<h3>⚠️ CONDICIONES OBLIGATORIAS DE LA FOTO</h3>
Para que el reconocedor funcione bien, la foto <b>tiene que cumplir todo esto</b>:
<ul>
<li>Sale <b>un solo perro</b>.</li>
<li>El perro está <b>de frente</b> y se ve <b>bien la cara</b>.</li>
<li>Se ve el <b>cuerpo entero</b> (o casi entero) y ocupa buena parte de la foto.</li>
<li><b>Sin personas</b> ni otros animales alrededor.</li>
<li>Foto <b>nítida y con buena luz</b> (nada borroso ni a contraluz).</li>
</ul>
<span class="warn">Si no se cumple, el resultado puede ser muy malo.</span>
</div>
""", unsafe_allow_html=True)
    ok = st.checkbox("✅ He comprobado que mi foto cumple todas las condiciones")
    if not ok:
        st.info("Marca la casilla para poder subir la foto.")
        return
    file = st.file_uploader("📷 Sube la foto del perro", type=["jpg", "jpeg", "png", "webp"])
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
    st.markdown("## 🏆 Las 3 razas que más se parecen")
    medals = ["🥇", "🥈", "🥉"]
    tags = ["la que más se parece", "segunda opción", "tercera opción"]
    html = ""
    for n, i in enumerate(np.argsort(-a["proba"])[:3]):
        name = R.prettyName(getModel()["breeds"][i]).title()
        css = "card first" if n == 0 else "card"
        html += '<div class="' + css + '"><div class="medal">' + medals[n] + '</div><div><div class="name">' + name + '</div><div class="tag">' + tags[n] + '</div></div></div>'
    st.markdown(html, unsafe_allow_html=True)
    if a["doubt"]:
        st.warning("⚠️ No he podido separar bien al perro del fondo, así que este resultado es poco fiable. Prueba con otra foto que cumpla las condiciones.")
    st.markdown('<p class="small">Solo conozco 120 razas. Si tu perro es mestizo o de otra raza, te enseño las que más se parecen. '
                'Mira la pestaña <b>Especificaciones</b> para ver paso a paso qué ha hecho el programa con tu foto.</p>', unsafe_allow_html=True)


# titulo de un paso
def stepTitle(text:str, new:bool=False) -> None:
    extra = '<span class="nuevo">no visto en clase</span>' if new else ""
    st.markdown('<div class="step">' + text + extra + '</div>', unsafe_allow_html=True)


# pestana 2: lo que hace el codigo con la foto, paso a paso
def showSpecs() -> None:
    st.markdown("## 🔬 Así funciona")
    st.write("Todo está hecho con técnicas clásicas de visión por computador, **sin deep learning**. "
             "Estos son los pasos que sigue el programa con cada foto.")
    a = st.session_state.get("analisis")
    if a is None:
        st.info("Sube una foto en la pestaña **Reconocer** y aquí verás, con tu foto, lo que hace el programa en cada paso.")
    s = a["steps"] if a is not None else None

    stepTitle("<b>Paso 1 · Buscar dónde está el perro</b> (saliency + Otsu + contornos)", True)
    st.write("El mapa de *saliency* marca lo que más llama la atención (en rojo). Con el umbral de **Otsu** (práctica 4.3) y "
             "`findContours` + `boundingRect` (práctica 4.1) se saca un rectángulo alrededor del perro. "
             "El *saliency* de OpenCV no se vio en clase.")
    if s is not None:
        c = st.columns(4)
        c[0].image(toRGB(s["small"]), caption="foto reducida")
        c[1].image(toRGB(s["saliency"]), caption="mapa de saliency")
        c[2].image(toRGB(s["thresh"]), caption="umbral de Otsu")
        c[3].image(toRGB(s["rect"]), caption="rectángulo del perro")

    stepTitle("<b>Paso 2 · Separar al perro del fondo</b> (GrabCut, práctica 4.2)")
    st.write("*GrabCut* recibe el rectángulo y decide, píxel a píxel, qué es perro y qué es fondo. "
             "El fondo se pone negro y el perro se recorta y se deja en **256×256** sin deformarlo. "
             "Si el resultado sale raro, se repite con otro rectángulo.")
    if s is not None:
        c = st.columns(3)
        c[0].image(toRGB(s["grabcut"]), caption="perro separado del fondo")
        c[1].image(toRGB(a["result"]), caption="recorte final 256×256")
        c[2].write("")

    stepTitle("<b>Paso 3 · Describir al perro con números</b> (7.108 números por foto)")
    st.write("Solo se mira dentro del perro (no el fondo negro). Cada descriptor mide una cosa distinta:")
    st.markdown("""
| Descriptor | Qué mide | Números | ¿Visto en clase? |
| --- | --- | --- | --- |
| Color HSV | los colores del pelo | 152 | Sí (prácticas 1.3 y 7.4) |
| HOG | la dirección de los bordes | 324 | Sí, en teoría (tema 6) |
| LBP (2 tamaños) | la textura del pelo | 1.536 | Solo nombrado |
| Bag of Words | puntos de interés SIFT agrupados con k-means | 1.000 | Sí (práctica 5.1, temas 5 y 7) |
| Fisher vector | SIFT + PCA + mezcla de gaussianas | 4.096 | **No** (extra) |
""")
    if s is not None:
        c = st.columns(3)
        c[0].image(toRGB(s["sift"]), caption="puntos donde se mira (SIFT denso)")
        c[1].image(toRGB(s["lbp"]), caption="textura (LBP)")
        c[2].caption("colores del perro (tono)")
        c[2].bar_chart(s["hue"])

    stepTitle("<b>Paso 4 · Decidir la raza</b> (regresión ridge)", True)
    st.write("Un clasificador aprendido con 11.751 fotos da una nota a cada una de las 120 razas. "
             "Se muestran las 3 razas con más nota. Se entrenó con la división oficial de *Stanford Dogs* (100 fotos por raza) "
             "y también con las fotos en espejo.")
    if a is not None:
        st.success("Para tu foto: " + " · ".join(R.prettyName(getModel()["breeds"][i]).title() for i in np.argsort(-a["proba"])[:3]))

    st.markdown("## 📊 Datos y resultados")
    st.markdown("""
- **Dataset:** *Stanford Dogs* (117 razas) y 3 razas de *Oxford-IIIT Pet* (shiba inu, havanese y american bulldog): **120 razas**.
- **Limpieza:** de 20.686 fotos se quitaron 452 donde el recorte falló mucho. Quedan **20.234**.
- **Resultado en test** (8.483 fotos que el modelo no vio): la raza correcta sale la **primera en el 31%** y **entre las 3 primeras en el 49%** (al azar sería 0,8% y 2,5%).
- **Con menos razas acierta más:** con 20 razas, 54% y 74%.
- **Qué ayudó más:** el Fisher vector (+5 puntos) y las fotos en espejo (+4 puntos en el top 3).

## ⚠️ Límites
- Solo conoce esas 120 razas. Con un mestizo o una raza que no sabe, enseña las que más se parecen.
- Funciona mejor con un solo perro, de frente y con la cara visible.
- En algunas razas acierta mucho (shiba inu 93%) y en otras muy poco (poodle estándar 2%).
""")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>🐶 ¿De qué raza es tu perro?</h1><p>Sube una foto y te digo a qué razas se parece</p></div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🐕 Reconocer", "🔬 Especificaciones"])
    with tab1:
        showRecognize()
    with tab2:
        showSpecs()


if __name__ == "__main__":
    main()
