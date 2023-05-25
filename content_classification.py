import streamlit as st
import numpy as np
import pickle
# from joblib import Parallel, delayed
import joblib
from streamlit_option_menu import option_menu
import pandas as pd


st.set_page_config(
   page_title="Content Classification")
# Load saved the model 

load_model = joblib.load(open('model_classification', 'rb'))

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
            options = ['Home', 'Content Classification']
        )

    
    if selected == 'Home':
        st.title(f'Overview Dataset')
        st.write('''
        Dataset yang digunakan merupakan hasil trace request dari website Wikipedia,
        yang diambil dari periode 19 September 2007 sampai dengan 2 Januari 2008. Selanjutnya
        hanya diambil 1036 baris dari total keseluruhan 10.628.126. yang dapat diakses pada
        [link berikut](http://www.wikibench.eu/wiki/2007-09/)
        ''')
        
        st.write('''
        Dataset Sebelum dilakukan preprocessing
        ''')

        df = pd.read_csv('https://raw.githubusercontent.com/hananlu/wikipedia/master/dataset/sampledata.csv', header=None, sep='\t', encoding='latin-1')
        df.drop(columns=4, inplace=True)
        df.columns = ['monotonic', 'timestamp', 'url', 'flag']

        st.dataframe(df.head())

        st.write('''
        Berikut merupakan dataset yang sudah dilakukan preprocessing, dari table dapat dilihat bahwa setiap url dipisah dengan '/' yang nantinya
        akan digunakan atribut pada model machine learning
        ''')

        st.write('''
        Dataset setelah dilakukan preprocessing
        ''')

        dataset = pd.read_csv('https://raw.githubusercontent.com/hananlu/wikipedia/master/dataset/wikipedia_popularity_link.csv')

        st.dataframe(dataset.head())

    if selected == 'Content Classification':
    #     #Giving a title

        st.title('Content Popularity Classification')

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
        
        if st.button('Content Classification Results'):
            diagnosis = car_prediction([prefix_1, prefix_2, prefix_3, prefix_4, prefix_5, prefix_6, prefix_7, prefix_8])


        st.success(diagnosis)

    


if __name__ == '__main__':
    main()
