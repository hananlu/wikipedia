import streamlit as st
import numpy as np
import pickle
from joblib import Parallel, delayed
import joblib


st.set_page_config(
   page_title="Content Prediction")
# Load saved the model 

load_model = joblib.load(open('/home/lutfianto/Documents/Exploratory Data/wikipedia/dataset/wikipedia_classification', 'rb'))

# Creating a function for prediction

def car_prediction(input_data):

    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = load_model.predict(input_data_reshaped)
    if prediction == 0:
        return 'Populer'
    else:
        return 'Unpopuler'


def main():

    #Giving a title

    st.title('Content Prediction')

    #Getting the input data from user

    prefix_1 = st.number_input('Prefix 1')
    prefix_2 = st.number_input('Prefix 2')
    prefix_3 = st.number_input('Prefix 3')
    prefix_4 = st.number_input('Prefix 4')
    prefix_5 = st.number_input('Prefix 5')
    prefix_6 = st.number_input('Prefix 6')
    prefix_7 = st.number_input('Prefix 7')
    prefix_8 = st.number_input('Prefix 8')

    # Code for Prediction
    diagnosis = ''

    # Creating a button for prediction
    
    if st.button('Content Prediction Results'):
        diagnosis = car_prediction([prefix_1, prefix_2, prefix_3, prefix_4, prefix_5, prefix_6, prefix_7, prefix_8])


    st.success(diagnosis)


if __name__ == '__main__':
    main()
