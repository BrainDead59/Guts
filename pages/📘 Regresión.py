import numpy as np   
import pandas as pd  
import seaborn as sns    
import streamlit as st                               
import matplotlib.pyplot as plt  

from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

st.set_page_config(
    page_title="Regresión",
    page_icon=":cyclone:",
    layout="wide",  
)

#Contenedor para el archivo de informacion fuente.
with st.container():
    st.markdown("# Regresion Logística")
    st.write("----")
    data = st.file_uploader("Escoge un archivo")

if data is None:
    st.session_state["data"] = "Inserta un archivo"
else:
    Informacion = pd.read_csv(data)
    Estado=0

    tab1,tab2=st.tabs(["Resumen de la informacion", "Aplicación del algoritmo"])

    #Resumen de la informacion
    with tab1:
        variables = st.text_input('Inserta las variables a eliminar, el nombre litera de la variable, separadas por coma y espacio: Var1, Var2, etc')

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("**Información insertada**"):
                st.dataframe(Informacion)

        #Calculo de las correlaciones
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
            st.write(" ")
            Estado=1
        else:
            variables = variables.split(", ")
            for col in variables:
                Informacion.pop(col)

            with col1:
                with st.expander("**Variables restantes**"):
                    st.dataframe(Informacion,use_container_width=True)
                    Estado=1

    #Aplicacion de los algoritmos
    with tab2:   
        col3, col4 = st.columns(2)
        if Estado==1:
            with col3:
                variablesPredict = st.text_input('Inserta las variables predictoras, el nombre literal de la variable, separadas por coma y espacio: Var1, Var2, etc')
            
            with col4:
                variablePredect = st.text_input('Inserta la variable clase')

            if len(variablesPredict)==0 and len(variablePredect)==0:
                st.write(" ")
            elif len(variablesPredict)!=0 and len(variablePredect)!=0:
                variablesPredict = variablesPredict.split(", ")
                variablePredect = variablePredect.split(", ")

                X = np.array(Informacion[variablesPredict])
                Y = np.array(Informacion[variablePredect])

                col5, col6, col7 = st.columns(3)

                with col5:
                    with st.expander("**Variables predictoras**"):
                        st.dataframe(X,use_container_width=True)

                with col6:
                    with st.expander("**Variable clase**"):
                        st.dataframe(Y,use_container_width=True)
                
                X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                    test_size = 0.2, 
                                                                                    random_state = 0,
                                                                                    shuffle = True)

                #Aplicacion del algoritmo
                ClasificacionRL = linear_model.LogisticRegression()
                ClasificacionRL.fit(X_train, Y_train)
                Y_ClasificacionRL = ClasificacionRL.predict(X_validation)

                with col7:
                    with st.expander("**Gráfica ROC**"):
                        CurvaROC = RocCurveDisplay.from_estimator(ClasificacionRL, X_validation, Y_validation)
                        grafica=plt.gcf()
                        st.pyplot(grafica)
                        st.write('**El valor de la precisión es:** '+str(accuracy_score(Y_validation, Y_ClasificacionRL)))
            
                col8, col9= st.columns(2)

                with col8:
                    valores = st.text_input('Inserta el valor de las variables predictoras, separadas por coma y espacio, en orden: Val1, Val2, etc')

                if len(valores)==0:
                    st.write(" ")
                else:
                    valores = valores.split(", ")
                    if len(valores)==len(variablesPredict):
                        datos = {}
                        contador=0
                        for i in valores:
                            if '.' in i:
                                datos[variablesPredict[contador]]=[float(i)]
                            else:
                                datos[variablesPredict[contador]]=[int(i)]
                            contador+=1

                        item = pd.DataFrame(datos)
                        Resultado=ClasificacionRL.predict(item)
                        with col8:
                            st.write('**El resultado de** '+str(variablePredect[0])+' **es:** '+str(Resultado[0]))


