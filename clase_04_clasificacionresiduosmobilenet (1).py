import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuración de la interfaz (formato ejecutivo y minimalista)
st.set_page_config(page_title="Clasificador de Residuos", layout="centered")

st.title("Clasificador de Residuos Sólidos")
st.markdown("Sistema de clasificación de residuos mediante redes neuronales convolucionales (MobileNetV2).")

# Diccionario de clases definido en el entrenamiento
CLASES = {
    0: "Vidrio (Glass)",
    1: "Papel (Paper)",
    2: "Plástico (Plastic)",
    3: "Cartón (Cardboard)",
    4: "Metal (Metal)",
    5: "Basura general (Trash)"
}

# Base de explicaciones técnicas por categoría
EXPLICACIONES = {
    0: "Material 100% reciclable. Debe disponerse libre de residuos orgánicos o líquidos para evitar contaminación cruzada en plantas de procesamiento.",
    1: "Reciclable si se mantiene seco y libre de grasas. El papel contaminado con alimentos pierde sus propiedades de reutilización.",
    2: "Polímero reciclable. Se recomienda enjuagar y comprimir el envase para optimizar la logística de almacenamiento y transporte.",
    3: "Altamente reciclable. Las cajas deben ser desarmadas y se recomienda retirar cintas adhesivas o grapas antes de su acopio.",
    4: "Material de alto valor de recuperación (acero/aluminio). Requiere enjuague básico para mitigar proliferación de vectores u olores.",
    5: "Residuos no reciclables en sistemas convencionales, material orgánico o elementos altamente contaminados que deben dirigirse a disposición final."
}

@st.cache_resource
def cargar_modelo():
    # Carga del modelo exportado en la Actividad 1
    # Asegúrese de que el archivo .keras se encuentre en la carpeta 'models'
    return tf.keras.models.load_model("models/garbage_classifier_mobilenetv2_pro.keras")

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"Error en la carga del modelo. Verifique la ruta del archivo .keras. Detalle: {e}")
    st.stop()

# Selección de método de entrada de imagen
metodo_entrada = st.radio(
    "Seleccione el método de captura de imagen:",
    ("Cargar archivo local", "Utilizar cámara del equipo")
)

imagen_fuente = None

if metodo_entrada == "Cargar archivo local":
    imagen_fuente = st.file_uploader("Seleccione una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])
else:
    imagen_fuente = st.camera_input("Capture una fotografía")

if imagen_fuente is not None:
    # Procesamiento de la imagen cargada
    imagen = Image.open(imagen_fuente).convert("RGB")
    st.image(imagen, caption="Imagen registrada para evaluación", use_column_width=True)
    
    st.markdown("---")
    st.write("Procesando clasificación...")
    
    # Preprocesamiento alineado con la configuración de MobileNetV2 (160x160 definido en el entrenamiento)
    imagen_procesada = imagen.resize((160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(imagen_procesada)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocesamiento nativo de MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Ejecución de la inferencia
    predicciones = modelo.predict(img_array)
    indice_predicho = np.argmax(predicciones[0])
    confiabilidad = np.max(predicciones[0]) * 100
    
    # Presentación de resultados
    st.subheader("Resultados del Análisis")
    st.write(f"**Categoría Identificada:** {CLASES[indice_predicho]}")
    st.write(f"**Nivel de Probabilidad:** {confiabilidad:.2f}%")
    
    # Explicación del resultado
    st.info(f"**Directriz de Manejo:** {EXPLICACIONES[indice_predicho]}")
    
