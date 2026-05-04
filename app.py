import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Image Classifier", layout="centered")

st.title("🧠 AI Image Classifier")
st.write("Upload an image and get prediction from your model")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = torch.load("best.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# IMAGE PROCESSING + PREDICTION
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # adjust if your model used different size
        transforms.ToTensor(),
    ])

    img_tensor = transform(image).unsqueeze(0)

    if st.button("Predict"):
        with torch.no_grad():
            output = model(img_tensor)

            # If classification
            if len(output.shape) > 1:
                pred = torch.argmax(output, dim=1).item()
            else:
                pred = output.item()

        st.success(f"Prediction: {pred}")
