import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Agri Text Analyzer")
st.title("🌾 Multilingual Text Detection App")

# -----------------------------
# Load YOLO
# -----------------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

model = load_yolo()

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    if st.button("Analyze"):

        # -----------------------------
        # YOLO Detection
        # -----------------------------
        st.subheader("🔍 Detection")
        results = model(image, conf=0.5, iou=0.4)

        result_img = results[0].plot()[:, :, ::-1]
        st.image(result_img)

        # -----------------------------
        # SAFE OCR LOAD (IMPORTANT)
        # -----------------------------
        st.subheader("📝 Extracting Text...")

        try:
            import easyocr
            reader = easyocr.Reader(['en','hi','te','ta'])
            ocr_results = reader.readtext(image)

            texts = []
            for (bbox, text, conf) in ocr_results:
                if conf > 0.4:
                    st.write(f"{text} ({conf:.2f})")
                    texts.append(text)

            # -----------------------------
            # Language Detection
            # -----------------------------
            st.subheader("🌐 Language Detection")

            from langdetect import detect

            for text in texts:
                try:
                    lang = detect(text)
                    st.write(f"{text} → {lang}")
                except:
                    st.write(f"{text} → Unknown")

        except Exception as e:
            st.error("❌ OCR not ready yet. Please refresh in 10–20 seconds.")

        st.success("✅ Done")
