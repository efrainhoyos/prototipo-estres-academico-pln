"""
Prototipo Web para la Deteccion de Estres Academico
Desarrollado con Streamlit, Integrando Pipeline PLN y Modelo BETO Fine-Tuned
Autores: Efrain Alfonso Hoyos Munoz, Yheison Jose Pino Acosta
"""

import streamlit as st
import torch
import sys
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException

# Agrega la carpeta src al path para importar el pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from preprocessing.pipeline import PipelinePLN

# Configura el modelo y las etiquetas
MODEL_PATH = "src/models/beto_finetuned"
LABELS = {0: "Positivo / Tranquilo", 1: "Neutro", 2: "Negativo / Estresado"}
COLORS = {0: "green", 1: "gray", 2: "red"}

# Diccionario de codigos de idioma a nombres completos en espanol
IDIOMAS_NOMBRES = {
    'af': 'afrikaans', 'ar': 'arabe', 'az': 'azerbaiyano', 'be': 'bielorruso',
    'bg': 'bulgaro', 'bn': 'bengali', 'ca': 'catalan', 'cs': 'checo',
    'cy': 'gales', 'da': 'danes', 'de': 'aleman', 'el': 'griego',
    'en': 'ingles', 'eo': 'esperanto', 'es': 'espanol', 'et': 'estonio',
    'eu': 'euskera', 'fa': 'persa', 'fi': 'finlandes', 'fr': 'frances',
    'ga': 'irlandes', 'gl': 'gallego', 'gu': 'guyarati', 'he': 'hebreo',
    'hi': 'hindi', 'hr': 'croata', 'hu': 'hungaro', 'hy': 'armenio',
    'id': 'indonesio', 'is': 'islandes', 'it': 'italiano', 'ja': 'japones',
    'ka': 'georgiano', 'kn': 'canares', 'ko': 'coreano', 'lt': 'lituano',
    'lv': 'leton', 'mk': 'macedonio', 'ml': 'malayalam', 'mn': 'mongol',
    'mr': 'marati', 'ms': 'malayo', 'my': 'birmano', 'ne': 'nepali',
    'nl': 'neerlandes', 'no': 'noruego', 'pa': 'punyabi', 'pl': 'polaco',
    'pt': 'portugues', 'ro': 'rumano', 'ru': 'ruso', 'sk': 'eslovaco',
    'sl': 'esloveno', 'so': 'somali', 'sq': 'albanes', 'sr': 'serbio',
    'sv': 'sueco', 'sw': 'suajili', 'ta': 'tamil', 'te': 'telugu',
    'th': 'tailandes', 'tl': 'tagalo', 'tr': 'turco', 'uk': 'ucraniano',
    'ur': 'urdu', 'vi': 'vietnamita', 'zh-cn': 'chino (simplificado)',
    'zh-tw': 'chino (tradicional)', 'la': 'latin', 'yi': 'yidis'
}

# Obtiene el nombre completo del idioma a partir de su codigo
def obtener_nombre_idioma(codigo):
    return IDIOMAS_NOMBRES.get(codigo, codigo)

# Aplica estilos CSS personalizados para traducir textos nativos y ajustar el layout
st.markdown("""
<style>
/* Traduce el texto del uploader de archivos */
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    visibility: hidden;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small::before {
    content: "200MB por archivo - CSV";
    visibility: visible;
    display: block;
}

/* Centra y agranda los titulos del sidebar */
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    text-align: center !important;
    font-size: 1.0rem !important;
    font-weight: bold !important;
    margin-top: 0.2rem !important;
    margin-bottom: 0.2rem !important;
    letter-spacing: 1px !important;
}

/* Ajusta el padding general del sidebar */
[data-testid="stSidebar"] {
    padding-top: 0.5rem !important;
}

[data-testid="stSidebar"] > div {
    padding-top: 0.2rem !important;
    padding-bottom: 0.2rem !important;
}

/* Ajusta el tamano de fuente del texto normal del sidebar */
[data-testid="stSidebar"] .stMarkdown {
    margin-top: 0.2rem !important;
    margin-bottom: 0.2rem !important;
}

/* Comprime botones del sidebar */
[data-testid="stSidebar"] button {
    margin-top: 0.2rem !important;
    margin-bottom: 0.2rem !important;
}

/* Comprime el uploader de archivos */
[data-testid="stFileUploader"] {
    margin-top: 0.2rem !important;
    margin-bottom: 0.2rem !important;
}

/* Comprime separadores */
[data-testid="stSidebar"] hr {
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
}
</style>
""", unsafe_allow_html=True)

# Configura la pagina con layout amplio para reducir margenes laterales
st.set_page_config(page_title="Detector de Estres Academico FUP", layout="wide")

# Carga el modelo BETO fine-tuned y su tokenizador
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer

# Carga el pipeline de preprocesamiento de PLN
@st.cache_resource
def load_pipeline():
    return PipelinePLN(idioma='es')

# Inicializa session_state con valores por defecto
if 'reset_counter' not in st.session_state:
    st.session_state['reset_counter'] = 0
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None
if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None
if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = None

# Muestra barra lateral con controles
with st.sidebar:
    # Muestra el titulo CONTROLES centrado y grande
    st.markdown("<h3>CONTROLES</h3>", unsafe_allow_html=True)

    # Reinicia la aplicacion y limpia texto, resultados y datos cargados
    if st.button("Nuevo Analisis", use_container_width=True):
        st.session_state['reset_counter'] += 1
        st.session_state['last_result'] = None
        st.session_state['uploaded_data'] = None
        st.session_state['batch_results'] = None
        st.rerun()

    st.markdown("---")

    # Muestra el titulo CARGAR ENCUESTAS centrado y grande
    st.markdown("<h3>CARGAR ENCUESTAS</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Subir archivo CSV (id, texto)",
        type=['csv'],
        help="El archivo debe tener columnas: id, texto"
    )

    # Procesa el archivo CSV cargado
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)

            # Valida que el archivo tenga las columnas correctas
            if 'id' not in df_uploaded.columns or 'texto' not in df_uploaded.columns:
                st.error("El archivo debe tener columnas 'id' y 'texto'")
            else:
                st.success(f"Archivo cargado: {len(df_uploaded)} encuestas")
                st.session_state['uploaded_data'] = df_uploaded

                # Muestra una vista previa de los datos cargados
                with st.expander("Ver datos cargados"):
                    st.dataframe(df_uploaded.head(10))
        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")

    st.markdown("---")

    # Muestra el titulo INFORMACION centrado y grande
    st.markdown("<h3>INFORMACION</h3>", unsafe_allow_html=True)
    st.markdown("""
**Proyecto:**  
Deteccion de Estres Academico

**Tecnologia:**  
BETO (BERT en espanol)

**Version:** 1.0

**Autores:**  
Efrain Alfonso Hoyos Munoz  
Yheison Pino

---

<div style="text-align: center;">
<b>Fundacion Universitaria de Popayan</b><br>
<b>2026</b>
</div>
""", unsafe_allow_html=True)

# Muestra titulo y descripcion principal de la aplicacion
st.title("Prototipo de Deteccion de Estres Academico")
st.markdown("""
Bienvenido al sistema de analisis de sentimientos basado en PLN. 
Ingrese un texto o cargue un archivo CSV para analizar el nivel de estres academico.
""")

# Carga recursos del sistema (modelo y pipeline)
with st.spinner("Inicializando modelo de IA y pipeline de lenguaje..."):
    model, tokenizer = load_model_and_tokenizer()
    pipeline = load_pipeline()

st.success("Sistema listo para analizar.")
st.divider()

# Crea area de entrada de texto con key dinamica para forzar reset al presionar Nuevo Analisis
text_area_key = f"input_text_area_{st.session_state['reset_counter']}"
user_text = st.text_area(
    "Escribe aqui tu texto o comentario academico:", 
    height=150, 
    placeholder="Ej: Estoy muy agobiado con los parciales y no puedo dormir...",
    key=text_area_key
)

# Crea botones de accion en dos columnas
col1, col2 = st.columns(2)

with col1:
    analizar_clicked = st.button("Analizar Sentimiento", type="primary", use_container_width=True)

with col2:
    # Limpia solo el resultado del analisis, manteniendo el texto ingresado
    if st.button("Limpiar Analisis", use_container_width=True):
        st.session_state['last_result'] = None
        st.rerun()

# Procesa el texto y muestra resultados cuando se hace clic en analizar
if analizar_clicked:
    # Valida que el texto no este vacio
    if not user_text.strip():
        st.warning("Por favor, ingresa un texto valido para analizar.")
    else:
        # Valida que el texto este en idioma espanol
        try:
            idioma_detectado = detect(user_text)

            # Rechaza textos que no esten en espanol
            if idioma_detectado != 'es':
                nombre_idioma = obtener_nombre_idioma(idioma_detectado)
                st.error(f"El texto no esta en espanol (idioma detectado: '{nombre_idioma}'). Por favor, ingresa un texto en espanol para realizar el analisis.")
                st.session_state['last_result'] = None
            else:
                # Ejecuta preprocesamiento del texto
                with st.spinner("Procesando texto..."):
                    resultado_pipeline = pipeline.procesar_texto_completo(user_text, return_steps=True)
                    processed_text = resultado_pipeline['texto_procesado']

                # Valida que el texto procesado tenga contenido significativo
                if not processed_text.strip():
                    st.warning("El texto no contiene palabras validas en espanol despues del preprocesamiento. Por favor, ingresa un texto con contenido significativo.")
                    st.session_state['last_result'] = None
                else:
                    # Tokeniza el texto para el modelo
                    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=128)

                    # Ejecuta prediccion del modelo
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        pred_label = torch.argmax(probs, dim=-1).item()
                        confidence = probs[0][pred_label].item()

                    # Guarda resultado en session_state
                    st.session_state['last_result'] = {
                        'texto_original': user_text,
                        'texto_procesado': processed_text,
                        'prediccion': pred_label,
                        'confianza': confidence,
                        'probabilidades': probs[0].tolist(),
                        'idioma': idioma_detectado
                    }
        except LangDetectException:
            # Maneja el caso en que no se puede detectar el idioma
            st.error("No se pudo detectar el idioma del texto. Por favor, ingresa un texto valido en espanol.")
            st.session_state['last_result'] = None

# Muestra resultado del analisis si existe
if st.session_state.get('last_result') is not None:
    result = st.session_state['last_result']
    label_text = LABELS[result['prediccion']]
    color = COLORS[result['prediccion']]

    st.markdown("### Resultado del Analisis")

    # Muestra sentimiento detectado con color
    st.markdown(
        f"**Sentimiento detectado:** <span style='color:{color}; font-size: 1.5em; font-weight: bold;'>{label_text}</span>", 
        unsafe_allow_html=True
    )

    # Muestra barra de confianza
    st.progress(result['confianza'])
    st.caption(f"Nivel de confianza del modelo: {result['confianza']:.2%}")

    # Muestra probabilidades de todas las clases
    with st.expander("Ver probabilidades detalladas"):
        st.markdown("**Distribucion de probabilidades por clase:**")
        for label_id, label_name in LABELS.items():
            prob = result['probabilidades'][label_id]
            st.markdown(f"- **{label_name}**: {prob*100:.2f}%")

    # Muestra detalles tecnicos del preprocesamiento
    with st.expander("Ver detalles tecnicos del preprocesamiento"):
        st.text(f"Texto original:\n{result['texto_original']}")
        st.text(f"Texto procesado (Lemas):\n{result['texto_procesado']}")

# Analiza datos cargados desde archivo CSV
if st.session_state.get('uploaded_data') is not None:
    st.divider()
    st.subheader("Analisis Masivo de Encuestas")

    df = st.session_state['uploaded_data']

    # Clasifica todas las encuestas del archivo cargado
    if st.button("Clasificar todas las encuestas", type="primary"):
        with st.spinner("Clasificando encuestas con el modelo BETO..."):
            resultados = []
            textos_no_espanol = []

            # Procesa cada encuesta del archivo
            for idx, row in df.iterrows():
                texto = row['texto']

                try:
                    idioma = detect(texto)

                    # Omite textos que no esten en espanol
                    if idioma != 'es':
                        textos_no_espanol.append(row['id'])
                        continue

                    # Preprocesa el texto
                    texto_procesado = pipeline.procesar_texto_completo(texto)['texto_procesado']

                    # Omite textos sin contenido significativo
                    if not texto_procesado.strip():
                        continue

                    # Tokeniza
                    inputs = tokenizer(texto_procesado, return_tensors="pt", truncation=True, max_length=128)

                    # Clasifica con el modelo
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        pred_label = torch.argmax(probs, dim=-1).item()
                        confidence = probs[0][pred_label].item()

                    # Agrega resultado a la lista
                    resultados.append({
                        'id': row['id'],
                        'texto': texto[:50] + "..." if len(texto) > 50 else texto,
                        'clasificacion': LABELS[pred_label],
                        'confianza': f"{confidence*100:.1f}%"
                    })
                except LangDetectException:
                    textos_no_espanol.append(row['id'])

            df_resultados = pd.DataFrame(resultados)
            st.session_state['batch_results'] = df_resultados

            # Muestra advertencia si se omitieron textos por no estar en espanol
            if textos_no_espanol:
                st.warning(f"Se omitieron {len(textos_no_espanol)} encuestas por no estar en espanol (IDs: {textos_no_espanol[:10]})")

    # Muestra resultados si existen
    if st.session_state.get('batch_results') is not None:
        st.markdown("### Resultados de la Clasificacion")
        st.dataframe(st.session_state['batch_results'], use_container_width=True)

        # Calcula estadisticas generales
        st.markdown("### Estadisticas Generales")

        col1, col2, col3 = st.columns(3)

        # Total de encuestas analizadas
        col1.metric("Total Analizadas", len(st.session_state['batch_results']))

        # Promedio de confianza
        avg_confianza = st.session_state['batch_results']['confianza'].apply(lambda x: float(x.replace('%', ''))).mean()
        col2.metric("Confianza Promedio", f"{avg_confianza:.1f}%")

        # Clase mas frecuente
        clase_frecuente = st.session_state['batch_results']['clasificacion'].mode()[0]
        col3.metric("Clase Mas Frecuente", clase_frecuente)

        # Muestra distribucion por clase
        st.markdown("### Distribucion por Clase")
        distribucion = st.session_state['batch_results']['clasificacion'].value_counts()

        for clase, cantidad in distribucion.items():
            porcentaje = (cantidad / len(st.session_state['batch_results'])) * 100
            st.markdown(f"- **{clase}**: {cantidad} encuestas ({porcentaje:.1f}%)")

        # Permite descargar los resultados en formato CSV
        csv = st.session_state['batch_results'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar resultados en CSV",
            data=csv,
            file_name='resultados_clasificacion.csv',
            mime='text/csv',
        )