import streamlit as st
import requests
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from pycaret.regression import load_model

menu = ['Prediction','Analytics']
selection = st.sidebar.selectbox("Medical Insurance Contracts", menu)

st.sidebar.write('Insurance Business.')

if selection== 'Prediction':
    st.title('IFRS 17 Insurance Contracts Classifier')
    #model = pickle.load(open(r"C:\Users\emutai\Desktop\GIT Stuff\Project\Claims_pred_RF.pkl", "rb"))
    model = load_model('Claims_pred_RF')
    minmax_scaler=MinMaxScaler()

#  def predict( AVG_AGE_SCALED, AVG_WEIGHT_SCALED, AVG_HEIGHT_SCALED, BASIC_PREMIUM_SCALED):
#     data = pd.DataFrame([[Age, Weight, Height, Premium]])
#     data.columns = ['AVG_AGE_SCALED', 'AVG_WEIGHT_SCALED', 'AVG_HEIGHT_SCALED', 'BASIC_PREMIUM_SCALED']
#     predictions = model.predict(data) 
#     return  int(predictions[0])

    def predict( AVG_AGE, AVG_WEIGHT, AVG_HEIGHT, BASIC_PREMIUM):
        data = pd.DataFrame([[Age, Weight, Height, Premium]])
        data.columns = ['AVG_AGE', 'AVG_WEIGHT', 'AVG_HEIGHT', 'BASIC_PREMIUM']
        predictions = model.predict(data) 
        return  int(predictions[0])

    Age= st.number_input("Enter the average age of the family[Years]")
    #Age=minmax_scaler.fit_transform([[Age]])
    Weight= st.number_input("Enter the average weight of the family[Kgs]")
    #Weight=minmax_scaler.fit_transform([[Weight]])
    Height= st.number_input("Enter the average height of the family members[Centimeters]")
    #Height=minmax_scaler.fit_transform([[Height]])
    Premium= st.number_input("Enter Quoted Premium[KES] ")
    #Premium=minmax_scaler.fit_transform([[Premium]])
    result = ""

# when 'Classify' is clicked, predict the contract to its classification
    if st.button("Classify"):
        result= predict(AVG_AGE=Age, AVG_WEIGHT=Weight, AVG_HEIGHT=Height, BASIC_PREMIUM=Premium)
        if result==0:
            rs='Entity is a non-onerous business'
        else:
            rs='Entity is an onerous business'
        st.success(f'{rs}')

if selection== 'Analytics':
    #df=pd.read_excel(r"C:\Users\emutai\Desktop\GIT Stuff\Project\Data.xlsx")
    df=sns.load_dataset('titanic')
    fig = plt.figure(figsize=(10, 4))
    sns.relplot(data=df,x='age',y='fare',hue='sex')
    st.pyplot(fig)
  
