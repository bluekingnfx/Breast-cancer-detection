import streamlit as st
import keras
import tensorflow as tf
import numpy as np
import os


# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image file")

if uploaded_file is not None:
  model_path = os.path.abspath("model.keras")
  if os.path.exists(model_path):
    model = keras.saving.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(50, 50))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array,batch_size=100,)
    if prediction[0][0] < 0.5:
      prediction = "Not cancerous"
    else:
      prediction = "Cancerous"

    st.success(prediction)
  else:
      raise FileNotFoundError(f"Model not found at {model_path}")
