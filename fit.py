import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción de estado fit ''')
st.image("imagenfit.jpg", caption="Descubre que tan fit eres.")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  age = st.number_input('Edad:', min_value=1, max_value=100, value = 1, step = 1)
  height_cm = st.number_input('Altura en cm:', min_value=0, max_value=200, value = 0, step = 1)
  weight_kg = st.number_input('Peso en kg:', min_value=0, max_value=150, value = 0, step = 1)
  heart_rate = st.number_input('Ritmo cardiaco:',min_value=0, max_value=200, value = 0, step = 1)
  blood_pressure = st.number_input('Presión arterial:', min_value=0, max_value=300, value = 0, step = 1)
  sleep_hours = st.number_input('Horas de sueño:', min_value=0, max_value=24, value = 0, step = 1)
  nutrition_quality = st.number_input('Calidad de nutrición (1 a 10):', min_value=1, max_value=10, value = 1, step = 1)
  activity_index = st.number_input('Nivel de actividad (1 a 5):', min_value=1, max_value=5, value =1, step = 1)
  smokes = st.number_input('Fumador (0: no) (1: si):', min_value=0, max_value=1, value = 0, step = 1)
  gender = st.number_input('Género (0: F) (1: M):', min_value=0, max_value=1, value = 0, step = 1)

  user_input_data = {'age': age,
                     'height_cm': height_cm,
                     'weight_kg': weight_kg,
                     'heart_rate': heart_rate,
                     'blood_pressure': blood_pressure,
                     'sleep_hours': sleep_hours,
                     'nutrition_quality': nutrition_quality,
                     'activity_index': activity_index,
                     'smokes': smokes,
                     'gender': gender}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

dfnuevo =  pd.read_csv('Fit.csv', encoding='latin-1')
X = dfnuevo.drop(columns='is_fit')
Y = dfnuevo['is_fit']

classifier = DecisionTreeClassifier(max_depth=6, criterion='entropy', min_samples_leaf=25, max_features=9, random_state=1613080)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No eres fit')
elif prediction == 1:
  st.write('Sí eres fit')
else:
  st.write('Sin predicción')
