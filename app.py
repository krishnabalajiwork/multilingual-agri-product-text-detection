import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Agri Text Detection")

st.title("🌾 Multilingual Agri Product Text Analyzer")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    if st.button("Detect"):
        results = model(image)

        # safer display (no cv2 issues)
        result_img = results[0].plot()[:, :, ::-1]
        st.image(result_img)

        st.write("Detections:")
        st.write(results[0].boxes)
