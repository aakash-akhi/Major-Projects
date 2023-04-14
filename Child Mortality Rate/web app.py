# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 00:52:39 2022

@author: savita
"""


import pickle
from sklearn.preprocessing import PolynomialFeatures 
import streamlit as st

# Load our pretrained model 

loaded_model= pickle.load(open('D:\ML & DS with Python\Source code\MAJOR Project/model.sav','rb'))

# creating a function for prediction
# Pass year as a parameter as we want to take year as an input

def prediciton(years):
    poly = PolynomialFeatures(degree=2)
    print("Prediction -Morality Rate Of World will be:",end=' ')
    return(loaded_model.predict(poly.fit_transform([[2020.5+years]])))


def main():
    
    # giving the title for the web app
    st.title('Child Mortality Rate Web App')
    
    #getting the input variable

    years= st.number_input('Number of years after 2020.5')
    
    # code from prediction
    
    # empty string for storing the whole result
    result=''
    
    #creating a button
    if st.button('Final Prediction'):
        result=prediciton(years)  # All the input data should be mentioned in the same order as needed by the model
        
    st.success(result)
    
# Run the main function     
if __name__=='__main__':
    main()

        
    