import streamlit as st
import numpy as np
import pickle
from joblib import Parallel, delayed
import joblib
from streamlit_option_menu import option_menu
import pandas as pd


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

    with st.sidebar:
        selected = option_menu(
            menu_title='Main Menu',
            options = ['Home', 'Content Prediction']
        )

    
    if selected == 'Home':
        st.title(f'Overview Dataset')
        st.write('''
        It was the best of times, it was the worst of times, it was
        the age of wisdom, it was the age of foolishness, it was
        the epoch of belief, it was the epoch of incredulity, it
        was the season of Light, it was the season of Darkness, it
        was the spring of hope, it was the winter of despair, (...)
        ''')

    if selected == 'Content Prediction':
        #Giving a title

        st.title('Content Prediction')

        #Getting the input data from user

        prefix_1 = st.text_input('Prefix 1')
        prefix_2 = st.text_input('Prefix 2')
        prefix_3 = st.text_input('Prefix 3')
        prefix_4 = st.text_input('Prefix 4')
        prefix_5 = st.text_input('Prefix 5')
        prefix_6 = st.text_input('Prefix 6')
        prefix_7 = st.text_input('Prefix 7')
        prefix_8 = st.text_input('Prefix 8')

        # Code for Prediction
        diagnosis = ''

        # Creating a button for prediction
        
        if st.button('Content Prediction Results'):
            diagnosis = car_prediction([prefix_1, prefix_2, prefix_3, prefix_4, prefix_5, prefix_6, prefix_7, prefix_8])


        st.success(diagnosis)

    


if __name__ == '__main__':
    main()
