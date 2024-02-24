import streamlit as st
import tensorflow as tf
import numpy as np


# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image file")

if uploaded_file is not None:
  model = tf.keras.models.load_model("model.keras")
  img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(50, 50))
  img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  prediction = model.predict(img_array,batch_size=100,)

  st.success(prediction)