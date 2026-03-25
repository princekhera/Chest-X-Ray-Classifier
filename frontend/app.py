import streamlit as st
import requests
from PIL import Image

st.title("Chest X-ray Classifier")

uploaded_file = st.file_uploader("Upload X-ray Image")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray")

    if st.button("Predict"):

        uploaded_file.seek(0)  # IMPORTANT

        response = requests.post(
            "http://127.0.0.1:5000/predict",
            files={"file": uploaded_file}
        )

        data = response.json()

        st.subheader("Prediction")

        # Get the class with the highest probability
        prediction = max(data, key=data.get)
        st.success(prediction)

        st.subheader("Probabilities")
        st.bar_chart(data)