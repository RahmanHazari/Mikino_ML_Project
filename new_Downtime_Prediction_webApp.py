# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 20:29:54 2023

@author: Admin
"""


import pandas as pd
import streamlit as st 
from sqlalchemy import create_engine
import joblib, pickle
from urllib.parse import quote 

imp_pipeline = joblib.load('processed1')  
out_pipeline = joblib.load('processed2')
sca_pipeline = joblib.load('processed3')
model = pickle.load(open('RFC_best.pkl', 'rb')) 


def fun1(data, user, pw, db):
    engine = create_engine(f'mysql+pymysql://{user}:%s@localhost:3306/{db}' % quote(f'{pw}'))
    data.drop(['Date', 'Machine_ID', 'Assembly_Line_No'], axis = 1,inplace=True)
    clean1 = pd.DataFrame(imp_pipeline.transform(data), columns = data.columns)
    clean2 = pd.DataFrame(out_pipeline.transform(clean1), columns = out_pipeline.get_feature_names_out())
    clean3 = pd.DataFrame(sca_pipeline.transform(clean2), columns = sca_pipeline.get_feature_names_out())
    prediction = pd.DataFrame(model.predict(clean3), columns = ['output'])
    final_data = pd.concat([prediction, data.iloc[:,:-1]], axis = 1)
    return final_data

def main():
    

    st.title("Makino Machine Downtime prediction")
    st.sidebar.title("Makino Machine Downtime prediction")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Makino Machine Downtime prediction </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame()
        
        
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = fun1(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))

if __name__=='__main__':
    main()
