import numpy as np    
import pandas as pd    
import streamlit as st           
import matplotlib.pyplot as plt   
 
from apyori import apriori

st.set_page_config(
    page_title="Apriori",
    page_icon=":cyclone:",
    layout="wide",  
    initial_sidebar_state="collapsed"
)

    #Contenedor para el archivo de informacion fuente.
with st.container():
    st.markdown("# Apriori")
    st.write("----")
    data = st.file_uploader("Escoge un archivo .csv")

if data is None:
    st.session_state["data"] = "Inserta un archivo"
else:
    DatosTransacciones = pd.read_csv(data, header=None)
    TransaccionesLista = DatosTransacciones.stack().groupby(level=0).apply(list).tolist()

    #Conteo de la frecuencia de elementos
    Transacciones = DatosTransacciones.values.reshape(-1).tolist()
    Lista = pd.DataFrame(Transacciones)
    Lista['Frecuencia'] = 1
    Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) 
    Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) 
    Lista = Lista.rename(columns={0 : 'Item'})

    tab1,tab2=st.tabs(["Resumen de la informacion", "Aplicación del algoritmo"])

    #Resumen de la informacion
    with tab1:
        
        col1, col2= st.columns(2)

        with col1:
            with st.expander("**Información insertada**"):
                st.dataframe(DatosTransacciones,use_container_width=True)

        with col2:
            with st.expander("**Información de la gráfica**"):
                st.dataframe(Lista,use_container_width=True)
        
        with st.expander("**Gráfica de la información**"):
            plt.figure(figsize=(16,20), dpi=300)
            plt.ylabel('Item')
            plt.xlabel('Frecuencia')
            plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
            grafica=plt.gcf()
            st.pyplot(grafica)

    #Aplicacion del algoritmo
    with tab2:
        col4, col5= st.columns(2)

        with col4:
            soporte = st.number_input('Inserta el soporte')
            confianza = st.number_input('Inserta la confianza')
            elevacion = st.number_input('Inserta la elevacion')

        if soporte <= 0.0 or soporte <= 0:
            st.write('El soporte debe ser mayor que cero')
        else:
            #Aplicacion del algoritmo
            ReglasC1 = apriori(TransaccionesLista, 
                        min_support=float(soporte), 
                        min_confidence=float(confianza), 
                        min_lift=float(elevacion))
            ResultadosC1 = list(ReglasC1)
            st.write('**Cantidad de reglas:** '+str(len(ResultadosC1)))

            with col5:
                reglas = st.number_input('Inserta la cantidad de reglas que quieras visualizar')

                with st.expander("**Reglas**"):
                    if reglas > len(ResultadosC1):
                        st.write('Inserta la misma cantidad de reglas creadas o menor')
                    elif reglas == 0:
                        st.write(' ')
                    else:
                        contador=0
                        for item in ResultadosC1:
                            if contador >= reglas:
                                break
                            else:
                                #El primer índice de la lista
                                Emparejar = item[0]
                                items = [x for x in Emparejar]
                                st.write("- **Regla:** " + str(item[0])+" **Soporte:** " + str(item[1])+" **Confianza:** " + str(item[2][0][2])+" **Lift:** " + str(item[2][0][3]))
                                contador+=1
                    