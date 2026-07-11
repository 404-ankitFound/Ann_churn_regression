import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
input_data = {}

st.title('Customer Churn Prediction')
input_data['CreditScore']=st.number_input('Credit Score',min_value=300,max_value=850,value=319)
input_data['Gender']=st.selectbox('Gender', ['Male', 'Female'])
input_data['Age']=st.number_input('Age',min_value=18,max_value=100,value=25)
input_data['Tenure']=st.number_input('Tenure',min_value=0,max_value=10,value=2)
input_data['Balance']=st.number_input('Balance',min_value=0.0,max_value=250000.0,value=0.0)
input_data['Geography']=st.selectbox('Geography', ['France', 'Spain', 'Germany'])



input_data['NumOfProducts']=st.number_input('Number of Products',min_value=0,max_value=4,value=0)
input_data['HasCrCard']=st.selectbox('Has Credit Card', [0, 1])
input_data['IsActiveMember']=st.selectbox('Is Active Member', [0, 1])
input_data['Exited']=st.selectbox('Exited', [0, 1])



# load trained model, scaller pickel and onehot encoded
model = load_model('model.h5')

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('X_scaler.pkl', 'rb') as f:
    X_scaler = pickle.load(f)
with open('y_scaler.pkl', 'rb') as f:
    y_scaler = pickle.load(f)

geo_enc=encoder.transform([[input_data['Geography']]])
geo_enc_df=pd.DataFrame(geo_enc, columns=encoder.get_feature_names_out(['Geography']))

input_df=pd.DataFrame([input_data])

# Drop original Geography column
input_df = input_df.drop('Geography', axis=1)

# Add encoded columns
input_df = pd.concat([input_df, geo_enc_df], axis=1)

input_df['Gender'] = input_df['Gender'].map({
    'Male': 1,
    'Female': 0
})

# scaling the input data

input_df = X_scaler.transform(input_df)




pred=model.predict(input_df)
pred=y_scaler.inverse_transform(pred)
st.write("Predicted Salary:", pred[0][0])
