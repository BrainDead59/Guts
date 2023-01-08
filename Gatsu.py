import streamlit as st
import base64

from PIL import Image
from streamlit_player import st_player

st.set_page_config(
    page_title="Gatsu",
    page_icon=":cyclone:",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

col1, col2=st.columns(2)
with col1:
    st.markdown("# Gatsu ")
    st.markdown("Gatsu es una herramienta que permite implementar los algoritmos vistos en clase de IA:")
    st.markdown("- Apriori\n - Regresión Logística\n - Métricas de Distancia\n - Clustering Jerárquico\n - Clustering Particional")
    st.markdown("Los algoritmos utilizan información que ya tiene el formato adecuado, en formato csv, cada algoritmo genera sus respectivas gráficas y tablas de contenido.")

with col2:
    imagenA = Image.open('recursos/D.png')
    st.image(imagenA,width=500)


st.markdown("---")

col3, col4, col5, col6=st.columns(4)

with col3:
    st.markdown("## Apriori ")
    st.markdown("Algoritmo de aprendizaje unsupervisado automático basado en reglas, que se utiliza para encontrar relaciones ocultas en los datos.")
    
with col3:
    imagenA = Image.open('recursos/B.png')
    st.image(imagenA,use_column_width=1)

with col4:
    st.markdown("## Clustering ")
    st.markdown("Algoritmos de aprendizaje unsupervisado que permite el agrupar los datos en base a las características que comparten.")
    
with col4:
    imagenA = Image.open('recursos/C.png')
    st.image(imagenA,use_column_width=1)

with col5:
    st.markdown("## Métricas de distancia ")
    st.markdown('Algoritmo que usa la medida de distancia como una puntuación objetiva que resume la diferencia entre dos elementos.')
    
with col5:
    imagenA = Image.open('recursos/A.png')
    st.image(imagenA,use_column_width=1)

with col6:
    st.markdown("## Regresión ")
    st.markdown("Algoritmo de aprendizaje supervisado cuyo objetivo es predecir valores binarios (0 o 1).")
    
with col6:
    imagenA = Image.open('recursos/E.png')
    st.image(imagenA,use_column_width=1)

st.markdown("---")
st.write("### Elaborado por: [Brain Dead](https://github.com/BrainDead59)")
st.write("Imágenes: https://storyset.com")

#py -m streamlit run Gatsu.py