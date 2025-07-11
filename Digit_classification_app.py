
import streamlit as st
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load your trained model
model = joblib.load("digit_classifier.pkl")  

st.title("Digit Classifier")

uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L').resize((28, 28))
        
        image_data = np.array(image)
        st.image(image, caption="Uploaded Digit", width=150)
        
        image_data_normalized=image_data/255.0
        image_data_reshaped = image_data.reshape(1, 28, 28)

# Predict

        prediction = model.predict(image_data_reshaped)
        prediction_label=np.argmax(prediction)
        st.write(f"### Predicted Digit: {prediction_label}")
