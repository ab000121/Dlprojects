
import streamlit as st
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


model = joblib.load("face_mask_classifier.pkl")  

 

st.title("Face Mask Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB').resize((128, 128))
        
        
        image_data = np.array(image)
        st.image(image, caption="Uploaded Image", width=150)
        
        image_data_normalized=image_data/255.0
        image_data_reshaped = np.reshape(image_data_normalized,(1, 128, 128,3))



# Predict

        prediction = model.predict(image_data_reshaped)
        prediction_label=np.argmax(prediction)
        if(prediction_label==0):
            st.write(f"### Predicted : Person is not wearing a mask")
        else:
              st.write(f"### Predicted: Person is wearing a mask")