import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Language Detection")
st.title("🌾 Language Detection (YOLO Model)")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# CLASS NAMES (EDIT THIS)
# -----------------------------
class_names = [
    "English",
    "Hindi",
    "Telugu",
    "Tamil",
    "Kannada",
    "Gujarati",
    "Malayalam",
    "Bengali"
]

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    if st.button("Detect Language"):

        results = model(image, conf=0.5, iou=0.4)

        st.subheader("Detected Languages:")

        boxes = results[0].boxes

        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)

            if conf > 0.5:
                lang = class_names[cls_id]

                st.write(f"🌐 {lang} ({conf:.2f})")

        # Show image with boxes
        result_img = results[0].plot()[:, :, ::-1]
        st.image(result_img, caption="Detection Result")
