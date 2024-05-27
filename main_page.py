import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Iris Classifier")
st.text("Provide URL of data for classification")

def load_model():
    version = 1
    parent_dir = os.getcwd()
    save_path=parent_dir+f"\\models\\iris_model\\{version}"
    model = tf.saved_model.load(save_path)
    return model


with st.spinner('Loading Model Into Memory....'):
    model = load_model()

classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']

path = st.text_input('Enter data URL to Classify.. ','https://storage.cloud.google.com/iris_mldeployment1/iris.csv')


if path is not None:

    parent_dir = os.getcwd()
    data=pd.read_csv(parent_dir+f"\\data\\raw\\iris.csv")

    print("Reading data completed...............")

    data=data.drop(['Id', 'Species'], axis=1)

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
        label =np.argmax(model.serve(data),axis=1)
    
    for l in label:
        st.write(classes[l])
    st.write("")
    




