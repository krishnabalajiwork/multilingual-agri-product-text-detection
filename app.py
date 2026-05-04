import streamlit as st
from PIL import Image
from ultralytics import YOLO
import easyocr
from langdetect import detect

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Agri Multilingual Text Analyzer")

st.title("🌾 Multilingual Agri Product Text Analyzer")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en','hi','te','ta','kn','ml','gu'])

model = load_yolo()
reader = load_ocr()

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):

        st.subheader("🔍 Step 1: Detect Text Regions (YOLO)")
        results = model(image, conf=0.5, iou=0.4)

        # Draw detection
        result_img = results[0].plot()[:, :, ::-1]
        st.image(result_img, caption="Detected Regions")

        st.subheader("📝 Step 2: Extract Text (OCR)")
        ocr_results = reader.readtext(image)

        texts = []

        for (bbox, text, conf) in ocr_results:
            if conf > 0.4:
                st.write(f"Text: {text} (Confidence: {conf:.2f})")
                texts.append(text)

        st.subheader("🌐 Step 3: Language Detection")

        for text in texts:
            try:
                lang = detect(text)
                st.write(f"🗣 '{text}' → Language: {lang}")
            except:
                st.write(f"🗣 '{text}' → Language: Unknown")

        st.success("✅ Analysis Complete")
