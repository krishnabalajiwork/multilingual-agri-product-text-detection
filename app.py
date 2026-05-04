import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# PAGE
# -----------------------------
st.set_page_config(page_title="Agri Text Detection", layout="centered")

st.title("🌾 Multilingual Agri Product Text Analyzer")

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")   # ✅ correct way
    return model

model = load_model()

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect"):
        results = model(image)

        # show result image with boxes
        result_img = results[0].plot()
        st.image(result_img, caption="Detection Result")

        # show raw detections
        st.write(results[0].boxes)
