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
selection = st.sidebar.selectbox("IFRS 17 Contracts", menu)

st.sidebar.write('Medical Insurance Contracts.')

if selection== 'Prediction':
    st.title('IFRS 17 Insurance Contracts Classifier')
    #model = pickle.load(open(r"C:\Users\emutai\Desktop\GIT Stuff\Project\Claims_pred_RF.pkl", "rb"))
    model = load_model('Claims_pred_RF')
    

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
    st.title('Features Data Distribution')
    claims_data=pd.read_excel(r'Data.xlsx')
    factor=2
    upper_lim=claims_data['BASIC_PREMIUM'].mean()+claims_data['BASIC_PREMIUM'].std()*factor
    lower_lim=claims_data['BASIC_PREMIUM'].mean()-claims_data['BASIC_PREMIUM'].std()*factor
    claims_data_model_valid=claims_data[(claims_data['BASIC_PREMIUM']>=lower_lim) & (claims_data['BASIC_PREMIUM']<=upper_lim)]
    col1, col2 = st.columns(2)
    #Removing outliers in BASIC_PREMIUM

    with col1: 
        fig = plt.figure(figsize=(10, 4))
        fig = px.histogram(claims_data_model_valid, x=["BASIC_PREMIUM"], template = 'plotly_dark', title = 'Paid Premium')
        #fig.show()
        st.plotly_chart(fig)
    with col2:
        fig = plt.figure(figsize=(10, 4))
        fig = px.histogram(claims_data_model_valid, x=["AVG_AGE"], template = 'plotly_dark', title = 'Average Age')
        #fig.show()
        st.plotly_chart(fig)
    col3, col4 = st.columns(2)
    with col3: 
        minmax_scaler=MinMaxScaler()
        claims_data_model_valid['CLAIMS_AMT_TRANSFORMED']=np.log(claims_data_model_valid['CLAIMS_AMT'])
        claims_data_model_valid['CLAIMS_AMT_TRANSFORMED']=minmax_scaler.fit_transform(claims_data_model_valid[['CLAIMS_AMT_TRANSFORMED']])
        fig = plt.figure(figsize=(10, 4))
        fig = px.histogram(claims_data_model_valid, x=["CLAIMS_AMT_TRANSFORMED"], template = 'plotly_dark', title = 'Claims')
        #fig.show()
        st.plotly_chart(fig)
    with col4:
        fig = plt.figure(figsize=(10, 4))
        fig = px.histogram(claims_data_model_valid, x=["AVG_HEIGHT"], template = 'plotly_dark', title = 'Average Height')
        #fig.show()
        st.plotly_chart(fig)