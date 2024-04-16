import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
np.random.seed(12345)
st.title("pagina")
datos= np.random.normal(0,1, size=(100,4))
data= pd.DataFrame(datos,columns= list("ASCD"))
st.dataframe(data)
E= np.random.normal(0,1, size =100)
y=data["A"]*2+data["S"]*3+data["C"]*4+data["D"]*0.3+10+E 
model=DecisionTreeRegressor(max_depth=4)
model.fit(data,y)
st.subheader("A")
val_A=st.slider("seleccione el valor de A",data["A"].min(), data["A"].max())
st.subheader("S")
val_S= st.slider("seleccione el valor de S",data["S"].min(), data["S"].max())
st.subheader("C")
val_C= st.slider("seleccione el valor de C",data["C"].min(), data["C"].max())
st.subheader("D")
val_D= st.slider("seleccione el valor de D",data["D"].min(), data["D"].max())
valores=np.array([[val_A, val_S, val_C, val_D]])
pre=model.predict([[val_A, val_S, val_C, val_D]])
st.write(pre)
