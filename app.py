import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("🌾 Language Detection from Product Text")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    if st.button("Detect Language"):

        results = model(image, conf=0.5, iou=0.4)

        try:
            import easyocr
            from langdetect import detect

            reader = easyocr.Reader(['en','hi','te','ta'])

            st.subheader("Detected Languages:")

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop detected region
                crop = image.crop((x1, y1, x2, y2))

                # OCR on cropped region
                ocr_result = reader.readtext(crop)

                for (_, text, conf) in ocr_result:
                    if conf > 0.4 and len(text.strip()) > 2:
                        try:
                            lang = detect(text)
                            st.write(f"🧾 '{text}' → 🌐 {lang}")
                        except:
                            st.write(f"🧾 '{text}' → Unknown")

        except:
            st.error("OCR not ready. Refresh once.")
