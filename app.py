import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Safe import for easyocr
try:
    import easyocr
except:
    easyocr = None

# Safe import for langdetect
try:
    from langdetect import detect
except:
    detect = None

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
# Load OCR safely
# -----------------------------
@st.cache_resource
def load_ocr():
    if easyocr is None:
        return None
    return easyocr.Reader(['en','hi','te','ta'])

reader = load_ocr()

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    if st.button("Analyze"):

        # -----------------------------
        # YOLO Detection
        # -----------------------------
        st.subheader("🔍 Detection")
        results = model(image, conf=0.5, iou=0.4)

        result_img = results[0].plot()[:, :, ::-1]
        st.image(result_img)

        # -----------------------------
        # OCR
        # -----------------------------
        if reader is None:
            st.error("❌ EasyOCR not installed properly")
        else:
            st.subheader("📝 Extracted Text")
            ocr_results = reader.readtext(image)

            texts = []
            for (bbox, text, conf) in ocr_results:
                if conf > 0.4:
                    st.write(f"{text} ({conf:.2f})")
                    texts.append(text)

            # -----------------------------
            # Language Detection
            # -----------------------------
            if detect is None:
                st.warning("⚠️ Language detection not available")
            else:
                st.subheader("🌐 Language Detection")
                for text in texts:
                    try:
                        lang = detect(text)
                        st.write(f"{text} → {lang}")
                    except:
                        st.write(f"{text} → Unknown")

        st.success("✅ Done")
