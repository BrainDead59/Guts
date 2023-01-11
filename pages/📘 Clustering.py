import numpy as np   
import pandas as pd    
import seaborn as sns 
import streamlit as st            
import matplotlib.pyplot as plt   
import scipy.cluster.hierarchy as shc

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

from kneed import KneeLocator
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances_argmin_min 

st.set_page_config(
    page_title="Clustering",
    page_icon=":cyclone:",
    layout="wide",  
    initial_sidebar_state="collapsed"
)

#Contenedor para el archivo de informacion fuente.
with st.container():
    st.markdown("# Clustering")
    st.write("----")
    data = st.file_uploader("Escoge un archivo")
    
if data is None:
    st.session_state["data"] = "Inserta un archivo"
else:
    InformacionJ = pd.read_csv(data) #Conserva la informacion despues de eliminar las columnas
    InformacionP = InformacionJ.copy()
    InformacionJerarquico = InformacionJ.copy() #Conserva la tabla original
    InformacionParticional = InformacionJ.copy()

    tab1,tab2,tab3=st.tabs(["Resumen de la informacion", "Aplicación del algoritmo Jerárquico","Aplicación del algoritmo Particional"])
    EstadoME=0
    EstadoJ=0
    EstadoP=0


    #Resumen de la informacion
    with tab1:
        variables = st.text_input('Inserta las variables a eliminar, el nombre literal de la variable, separadas por coma y espacio: Var1, Var2, etc')

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("**Información insertada**"):
                st.dataframe(InformacionJ)

        #Calculo de las correlaciones
        with col2:
            with st.expander("**Matriz de correlaciones**"):
                CorrInformacionJ = InformacionJ.corr(method='pearson')
                plt.figure(figsize=(14,7))
                MatrizInf = np.triu(CorrInformacionJ)
                sns.heatmap(CorrInformacionJ, cmap='RdBu_r', annot=True, mask=MatrizInf)
                grafica=plt.gcf()
                st.pyplot(grafica)

        #Estandarizacion de los datos
        if len(variables)==0:
            st.write(" ")
        elif variables=="Ninguna":
            with col1:
                with st.expander("**Datos estandarizados**"):
                    estandarizar = StandardScaler()                                
                    MEstandarizada = estandarizar.fit_transform(InformacionJ)   
                    st.dataframe(MEstandarizada,use_container_width=True)
                    EstadoME=1
        else:
            variables = variables.split(", ")
            for col in variables:
                InformacionJ.pop(col)
            InformacionP=InformacionJ.copy()

            with col1:
                with st.expander("**Variables restantes**"):
                    st.dataframe(InformacionJ,use_container_width=True)
            
            with col2:
                with st.expander("**Datos estandarizados**"):
                    estandarizar = StandardScaler()                                
                    MEstandarizada = estandarizar.fit_transform(InformacionJ)   
                    st.dataframe(MEstandarizada,use_container_width=True)
                    EstadoME=1
    
    #Aplicacion del algoritmo de clustering Jerarquico
    with tab2:
        col3, col4 = st.columns(2)

        with col3:
            clusters = st.number_input('Inserta el numero de clusters')

        with col3:
            if EstadoME==1:
                if clusters>0:
                    with st.expander("**Clusters**"):
                        MJerarquico = AgglomerativeClustering(n_clusters=int(clusters), linkage='complete', affinity='euclidean')
                        MJerarquico.fit_predict(MEstandarizada)

                        InformacionJerarquico['clusterJ'] = MJerarquico.labels_
                        InformacionJ['clusterJ'] = MJerarquico.labels_
                        st.dataframe(InformacionJerarquico,use_container_width=True)
                        EstadoJ=2
                else:
                    st.write("El numero de clusters debe ser positivo")
        
        with col4:
            with st.expander("**Grafica de los cluster**"):
                if EstadoJ==2:
                    plt.figure(figsize=(10, 7))
                    plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
                    plt.grid()
                    grafica=plt.gcf()
                    st.pyplot(grafica)

        with col4:
            with st.expander("**Centroides de cada cluster**"):
                if EstadoJ==2:
                    CentroidesJ = InformacionJ.groupby(['clusterJ']).mean()
                    st.dataframe(CentroidesJ,use_container_width=True)
        
        with col3:
            with st.expander("**Elementos del Cluster**"):
                if EstadoJ==2:
                    clusterJ = st.number_input('Inserta el numero del cluster que deseas consultar, con indice de 0 a n-1')
                    if clusterJ>=0 and clusterJ<clusters:
                        st.dataframe(InformacionJerarquico[InformacionJerarquico.clusterJ == int(clusterJ)],use_container_width=True)
                        st.write('**Cantidad de elementos en el cluster:** '+str(InformacionJerarquico[InformacionJerarquico.clusterJ == int(clusterJ)].count()[0]))
                    else:
                        st.write("Inserta una cantidad dentro del rango de clusters creados")

        with st.expander("**Árbol**"):
            if EstadoJ==2:
                plt.figure(figsize=(10, 7))
                plt.xlabel('Observaciones')
                plt.ylabel('Distancia')
                Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean')) 
                grafica=plt.gcf()
                st.pyplot(grafica)

    #Aplicacion del algoritmo de clustering Particional
    with tab3:
        if EstadoME==1:
            col5,col6=st.columns(2)

            with col5:
                inicial = st.number_input('Inserta la cantidad de clusters iniciales')

            with col6:
                final = st.number_input('Inserta la cantidad de clusters finales')

            if (final>0 and inicial>0) and (inicial <= final) and ((final-1-inicial)>=2):
                SSE = []
                for i in range(int(inicial), int(final)-1):
                    km = KMeans(n_clusters=i, random_state=0, n_init=int(inicial))
                    km.fit(MEstandarizada)
                    SSE.append(km.inertia_)

                kl = KneeLocator(range(int(inicial),int(final)-1), SSE, curve="convex", direction="decreasing")

                with col5:
                    with st.expander("**Clusters**"):
                        if kl.elbow:
                            MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
                            MParticional.predict(MEstandarizada)
                            InformacionParticional['clusterP'] = MParticional.labels_
                            InformacionP['clusterP'] = MParticional.labels_
                            st.dataframe(InformacionParticional,use_container_width=True)
                            EstadoP=2

                #Aplicacion del algoritmo del codo
                with col6:
                    with st.expander("**Elbow Method**"):
                        if EstadoP==2:
                            plt.figure(figsize=(10, 7))
                            plt.plot(range(int(inicial), int(final)-1), SSE, marker='o')
                            plt.xlabel('Cantidad de clusters *k*')
                            plt.ylabel('SSE')
                            plt.title('Elbow Method')
                            grafica=plt.gcf()
                            st.pyplot(grafica)
                            st.write('**Número de clusters:** '+str(kl.elbow))

                with col6:
                    with st.expander("**Gráfica de los cluster**"):
                        if EstadoP==2:
                            plt.figure(figsize=(10, 7))
                            plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MParticional.labels_)
                            plt.grid()
                            grafica=plt.gcf()
                            st.pyplot(grafica)

                with col5:
                    with st.expander("**Elementos del Cluster**"):
                        if EstadoP==2:
                            clusterP = st.number_input('Inserta el numero del cluster que deseas consultar, con indice de 0 a n-1 ')
                            if clusterP>=0 and clusterP<kl.elbow:
                                st.dataframe(InformacionParticional[InformacionParticional.clusterP == int(clusterP)],use_container_width=True)
                                st.write('**Cantidad de elementos en el cluster:** '+str(InformacionParticional[InformacionParticional.clusterP == int(clusterP)].count()[0]))
                            else:
                                st.write("Inserta una cantidad dentro del rango de clusters creados")

                with col6:
                    with st.expander("**Centroides de cada cluster**"):
                        if EstadoP==2:
                            CentroidesP = InformacionP.groupby(['clusterP']).mean()
                            st.dataframe(CentroidesP,use_container_width=True)