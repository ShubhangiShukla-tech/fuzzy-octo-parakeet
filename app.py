
import streamlit as st
from PIL import Image
import pickle
from preprocess.preprocess import preprocess_image

st.title("🧪 Wafer Defect Detection App")
st.write("Upload a wafer map image to detect if it's a PASS or FAIL.")

uploaded_file = st.file_uploader("Choose a wafer image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Wafer Map", use_column_width=True)

    with st.spinner("Analyzing..."):
        features = preprocess_image(image)

        with open("model/model.pkl", "rb") as f:
            model = pickle.load(f)

        prediction = model.predict(features)[0]
        label = "PASS ✅" if prediction == 1 else "FAIL ❌"
        st.success(f"Prediction: **{label}**")
