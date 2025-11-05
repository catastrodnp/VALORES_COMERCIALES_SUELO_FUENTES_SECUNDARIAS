# Configurar la aplicación de Streamlit (debe ser la primera línea)
import streamlit as st
st.set_page_config(layout="wide", page_title="Análisis Inmobiliario", page_icon="🏠")
# Configuración del logo y el título

st.markdown(
    """
    <style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .main-logo {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        margin-left: 10px;
    }
    .main-title {
        text-align: left;
        font-size: 2.4em; /* Tamaño del título aumentado */
        color: #1A237E; /* Azul oscuro */
        margin: 0;
    }
    </style>
    <div class="header-container">
        <h1 class="main-title">Herramienta de Análisis Inmobiliario</h1>
        <div class="main-logo">
            <img src="https://www.dnp.gov.co/img/og-img.jpg" width="250"> <!-- Logo más grande -->
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# CSS personalizado
st.markdown(
    """
    <style>
    /* Centrar las pestañas y personalizar el estilo */
    div[data-testid="stHorizontalBlock"] {
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }

    /* Cambiar tamaño y color de la fuente de las pestañas */
    div[data-testid="stHorizontalBlock"] button {
        font-size: 1.2em; /* Tamaño de fuente más grande */
        background-color: #1A237E; /* Azul oscuro */
        color: white;
        border-radius: 5px;
        margin: 0 5px;
        padding: 10px;
    }

    /* Efecto hover en las pestañas */
    div[data-testid="stHorizontalBlock"] button:hover {
        background-color: #3949AB; /* Azul más claro */
    }


    button[data-baseweb="button"]:hover {
        background-color: #3949AB; /* Azul más claro */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Importar las demás librerías necesarias
import numpy as np
import pandas as pd
import pickle
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import os

# Paso 1: Cargar el modelo entrenado desde el archivo pkl
@st.cache_resource
def cargar_modelo(ruta):
    with open(ruta, 'rb') as file:
        modelo = pickle.load(file)
    return modelo

# Cargar el modelo utilizando la nueva función con st.cache_resource
model = cargar_modelo(os.path.join('model','ValoresSuelo_Xgboost_Col_app_03122024.pkl'))


tabs = st.tabs(["1 - Predicción de Precios", "2 - Mapas de Calor y Clusters", "3 -Interpolación - Zonas"])


# Diccionarios para mapear los nombres
estado_inmueble_map = {
    "Usado": 116008, "Sin especificar": 78188, "Bueno": 20246, "Excelente": 18623,
    "Remodelado": 843, "Nuevo": 335, "Obra gris": 1
}
tipo_inmueble_map = {
    "apartamento": 219524, "casa": 85201, "lote": 21311, "oficina": 12057,
    "edificio": 10778, "local": 8599, "finca": 7420, "bodega": 6985,
    "apartaestudio": 5351, "parqueadero": 4628, "casa campestre": 3969,
    "proyecto": 2876, "penthouse": 2002, "otros": 1253, "habitacion": 634,
    "consultorio": 619, "hotel": 327, "cabaña": 273, "casalote": 176,
    "hacienda": 99, "almacen": 91, "estacion": 43, "garage": 29,
    "clinica": 11, "hangar": 9
}
# --- Pestaña 1: Predicción de Precios ---
with tabs[0]:
    st.markdown(
    """
    <h2 style="font-size: 1.5em; color: #000;">Predicción de Precio de Inmuebles en Colombia</h2>
    """, 
    unsafe_allow_html=True)


    # Sidebar con descripción e instrucciones
    st.sidebar.markdown(
        """
        ### Descripción
        Bienvenido a la **Herramienta de Análisis Inmobiliario**, una solución interactiva basada en mapas y modelos de aprendizaje automático diseñada para facilitar el análisis del mercado inmobiliario en Colombia. Esta herramienta cuenta con tres módulos principales:

        1. **Predicción de Precios**: Utiliza un modelo entrenado con **XGBoost** y más de **350,000 datos reales** del mercado inmobiliario colombiano para estimar el precio de propiedades basándose en características específicas y ubicación.
        
        2. **Mapas de Calor y Clusters**: Ofrece una visualización interactiva de datos reales del mercado inmobiliario, permitiendo explorar tendencias de precios, áreas de mayor densidad y patrones geográficos clave.

        3. **Interpolación - Zonas**: Genera zonificaciones por departamentos basadas en precios del suelo utilizando diagramas de Voronoi, brindando una visión detallada del comportamiento del mercado por región.

        ### Instrucciones
        - Navegue entre los módulos utilizando las pestañas superiores.
        - Explore precios, mapas y visualizaciones de manera interactiva.
        - Seleccione características de propiedades y ubicaciones específicas para predicciones personalizadas.
        """
    )

    # Crear widgets para la entrada de datos en 2 columnas y 4 filas
    col1, col2 = st.columns(2)
    with col1:
        area_total = st.number_input('Área Total (m²):', min_value=1, value=120)
        antiguedad = st.number_input('Antigüedad (años):', min_value=0, value=10)
        habitaciones = st.number_input('Número de Habitaciones:', min_value=0, value=3, step=1)
        tipo_inmueble = st.selectbox('Tipo de Inmueble:', list(tipo_inmueble_map.keys()))
    with col2:
        area_construida = st.number_input('Área Construida (m²):', min_value=1, value=100)
        estrato = st.selectbox('Estrato:', [1, 2, 3, 4, 5, 6])
        banos = st.number_input('Número de Baños:', min_value=0, value=2, step=1)
        estado_inmueble = st.selectbox('Estado del Inmueble:', list(estado_inmueble_map.keys()))


    # Mapa para seleccionar coordenadas
    col_map, col_info = st.columns([3, 1])
    with col_map:
        st.write('Selecciona un punto en el mapa para obtener las coordenadas:')
        mapa = folium.Map(location=[4.570868, -74.297333], zoom_start=6)
        map_data = st_folium(mapa, height=400, width=700)

    latitud, longitud, ubicacion = None, None, None
    if map_data['last_clicked'] is not None:
        latitud = map_data['last_clicked']['lat']
        longitud = map_data['last_clicked']['lng']
        geolocator = Nominatim(user_agent="SS")
        location = geolocator.reverse((latitud, longitud), exactly_one=True)
        ubicacion = location.address if location else "Ubicación desconocida"

        with col_info:
            st.write(f"**Coordenadas:** Latitud: {latitud:.6f}, Longitud: {longitud:.6f}")
            st.write(f"**Ubicación:** {ubicacion}")

    # Función de predicción
    def predecir_precio(data):
        try:
            log_precio = model.predict(data)[0]
            precio = np.exp(log_precio)
            return precio
        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")
            return None

    # Realizar la predicción
    with col_info:
        if st.button('Realizar Predicción'):
            if latitud is None or longitud is None:
                st.warning('Por favor, selecciona un punto en el mapa.')
            else:
                # Convertir nombres a códigos
                tipo_inmueble_encoded = tipo_inmueble_map[tipo_inmueble]
                estado_inmueble_encoded = estado_inmueble_map[estado_inmueble]

                data = pd.DataFrame({
                    'estrato': [estrato], 'habitaciones': [habitaciones], 'baños': [banos],
                    'area_construida': [area_construida], 'latitud': [latitud],
                    'longitud': [longitud], 'antigüedad': [antiguedad],
                    'area_total': [area_total],
                    'tipo_inmueble_encoded': [tipo_inmueble_encoded],
                    'estado_inmueble_encoded': [estado_inmueble_encoded]
                })
                precio_predicho = predecir_precio(data)
                if precio_predicho is not None:
                    st.success(f"El precio predicho del inmueble es: {precio_predicho:,.2f} COP")
                    st.balloons()
                else:
                    st.error("No se pudo realizar la predicción. Verifica los datos ingresados.")


# --- Pestaña 2: Mapas (Lista desplegable) ---
with tabs[1]:
    st.markdown(
        """
        <h2 style="font-size: 1.5em; color: #000;">Mapas de valores del suelo</h2>
        """, 
        unsafe_allow_html=True
    )
    st.sidebar.markdown("### Mapas de valores del suelo")
    
    # Ruta de la carpeta que contiene los archivos HTML para los mapas
    carpeta_proyectos = os.path.join('data','mapas')
    html_files_proyectos = [f for f in os.listdir(carpeta_proyectos) if f.endswith('.html')]
    
    # Eliminar la extensión '.html' para mostrar nombres amigables en la lista desplegable
    html_files_proyectos_no_ext = [os.path.splitext(f)[0] for f in html_files_proyectos]
    
    # Crear una lista desplegable para seleccionar el mapa
    selected_mapa = st.selectbox("Selecciona un mapa para visualizar:", html_files_proyectos_no_ext)
    
    # Mostrar el contenido del archivo HTML seleccionado
    if selected_mapa:
        html_path_proyecto = os.path.join(carpeta_proyectos, selected_mapa + '.html')
        with open(html_path_proyecto, 'r', encoding='utf-8') as file:
            html_content_proyecto = file.read()
        st.components.v1.html(html_content_proyecto, height=600, scrolling=True)


# --- Pestaña 3: Intepolación por Departamentos (Lista desplegable) ---
with tabs[2]:
    st.markdown(
    """
    <h2 style="font-size: 1.5em; color: #000;">Interpolación y zonficación de precios por Departamentos
    """, 
    unsafe_allow_html=True)
    
    # Ruta de la carpeta que contiene los archivos HTML para la pestaña 2
    #carpeta_departamentos = r'C:\Users\Sebastian\Documents\ComponenteEconomico\Interpolacion\Voronoi_5_2000'
    carpeta_departamentos=os.path.join('data','interpolacion')
    html_files_departamentos = [f for f in os.listdir(carpeta_departamentos) if f.endswith('.html')]
    
    # Eliminar el sufijo '.html' de los nombres para la lista desplegable
    html_files_departamentos_no_ext = [os.path.splitext(f)[0] for f in html_files_departamentos]
    
    # Dropdown para seleccionar un archivo HTML desde la carpeta de departamentos
    selected_departamento = st.selectbox("Selecciona un departamento para visualizar:", html_files_departamentos_no_ext)
    
    # Mostrar contenido del archivo HTML seleccionado
    if selected_departamento:
        html_path_departamento = os.path.join(carpeta_departamentos, selected_departamento + '.html')
        with open(html_path_departamento, 'r', encoding='utf-8') as file:
            html_content_departamento = file.read()
        st.components.v1.html(html_content_departamento, height=600, scrolling=True)



st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
    <small>Desarrollado por Sebastián Salazar Galeano | © 2024</small>
    </div>
    """,
    unsafe_allow_html=True
)

