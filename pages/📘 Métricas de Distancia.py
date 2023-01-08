import numpy as np   
import pandas as pd  
import seaborn as sns    
import streamlit as st                               
import matplotlib.pyplot as plt        

from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

st.set_page_config(
    page_title="Métricas de Distancia",
    page_icon=":cyclone:",
    layout="wide",  
)

with st.container():
    st.markdown("# Métricas de Distancia")
    st.write("----")
    data = st.file_uploader("Escoge un archivo .csv")

if data is None:
    st.session_state["data"] = "Inserta un archivo"
else:
    Informacion = pd.read_csv(data)
    EstadoME=0

    tab1,tab2=st.tabs(["Resumen de la informacion", "Aplicación de los algoritmos"])

    with tab1:
        variables = st.text_input('Inserta las variables a eliminar, el nombre literal de la variable, separadas por coma y espacio: Var1, Var2, etc')

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("**Información insertada**"):
                st.dataframe(Informacion)

        with col2:
            with st.expander("**Matriz de correlaciones**"):
                CorrInformacion = Informacion.corr(method='pearson')
                plt.figure(figsize=(14,7))
                MatrizInf = np.triu(CorrInformacion)
                sns.heatmap(CorrInformacion, cmap='RdBu_r', annot=True, mask=MatrizInf)
                grafica=plt.gcf()
                st.pyplot(grafica)
        
        if len(variables)==0:
            st.write(" ")
        elif variables=="Ninguna":
            with col1:
                with st.expander("**Datos estandarizados**"):
                    estandarizar = StandardScaler()                                
                    MEstandarizada = estandarizar.fit_transform(Informacion)   
                    st.dataframe(MEstandarizada,use_container_width=True)
                    EstadoME=1
        else:
            variables = variables.split(", ")
            for col in variables:
                Informacion.pop(col)

            with col1:
                with st.expander("**Variables restantes**"):
                    st.dataframe(Informacion,use_container_width=True)
            
            with col2:
                with st.expander("**Datos estandarizados**"):
                    estandarizar = StandardScaler()                                
                    MEstandarizada = estandarizar.fit_transform(Informacion)   
                    st.dataframe(MEstandarizada,use_container_width=True)
                    EstadoME=1
        
    with tab2:
        if EstadoME==1:
            col1, col2, col3 = st.columns(3)

            with col1:
                elementoUno = st.number_input('Inserta la posicion del primer elemento')

            with col2:
                elementoDos = st.number_input('Inserta la posicion del segundo elemento')

            with col3:
                tipoDistancia = st.text_input('Inserta el tipo de distancia a calcular')

            if (elementoDos and elementoUno)>=0 and (elementoDos<len(Informacion) and elementoUno<len(Informacion)) and len(tipoDistancia)!=0:
                Objeto1 = MEstandarizada[int(elementoUno)]
                Objeto2 = MEstandarizada[int(elementoDos)]
                distancia = 0.0
                if tipoDistancia=="Euclideana":
                    distancia = distance.euclidean(Objeto1,Objeto2)
                elif tipoDistancia=="Chebyshev":
                    distancia = distance.chebyshev(Objeto1,Objeto2)
                elif tipoDistancia=="Manhattan":
                    distancia = distance.cityblock(Objeto1,Objeto2)
                elif tipoDistancia=="Minkowski":
                    distancia = distance.minkowski(Objeto1,Objeto2,p=1.5)
                st.write('**La distancia entre los elementos es:** '+str(distancia))

        col4, col5= st.columns(2)

        with col4:
            with st.expander("**Matriz de distancias - Euclideana**"):
                if EstadoME==1:
                    DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean') 
                    MEuclidiana = pd.DataFrame(DstEuclidiana)
                    st.dataframe(MEuclidiana,use_container_width=True)

        with col4:
            with st.expander("**Matriz de distancias - Chebyshev**"):
                if EstadoME==1:
                    DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
                    MChebyshev = pd.DataFrame(DstChebyshev)
                    st.dataframe(MChebyshev,use_container_width=True)

        with col5:
            with st.expander("**Matriz de distancias - Manhattan**"):
                if EstadoME==1:
                    DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
                    MManhattan = pd.DataFrame(DstManhattan)
                    st.dataframe(MManhattan,use_container_width=True)

        with col5:
            with st.expander("**Matriz de distancias - Minkowski**"):
                if EstadoME==1:
                    DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
                    MMinkowski = pd.DataFrame(DstMinkowski)
                    st.dataframe(MMinkowski,use_container_width=True)