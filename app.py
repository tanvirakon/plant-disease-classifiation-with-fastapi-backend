# frontend
import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="plant disease", page_icon="📝", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #212739;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# plant disease")

# 2. Add the file uploader widget
uploaded_file = st.file_uploader(
    label="Choose an plant image...", type=["png", "jpg", "jpeg"]
)

# 3. Process and display the image if a file is uploaded
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing leaf disease..."):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }
        backend_url = "http://localhost:8000/predict"
        response = requests.post(backend_url, files=files)

        if response.status_code == 200:
            result = response.json()
            disease_name = result["class"].replace("___", " - ").replace("_", " ")
            confidence_pct = result["confidence"] * 100

            st.success(f"**Predicted:** {disease_name}")
            st.metric(label="Confidence", value=f"{confidence_pct:.2f}%")
            st.progress(result["confidence"])
        else:
            st.error("Prediction failed. Check backend connection.")
