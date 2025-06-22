import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# -------------------- CONFIG --------------------
IMG_SIZE = (150, 150)
MODEL_PATH = "model/cnn_intel_model.h5"  # or cnn_intel_model.keras
CONFUSION_MATRIX_PATH = "outputs/confusion_matrix.png"
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# -------------------------------------------------

# Set full-page layout
st.set_page_config(page_title="Scene Classifier Dashboard", layout="wide")

# Load model (cached to avoid reload on every run)
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ----------------- Streamlit App -----------------

st.title("🌄 Scene Image Classifier - Intel Dataset")

# Use tabs to separate upload and evaluation sections
tab1, tab2 = st.tabs(["📤 Upload Image", "📊 Evaluation Dashboard"])

# ------------- Tab 1: Upload & Predict -------------
with tab1:
    st.header("📥 Upload an Image")
    uploaded_file = st.file_uploader("Choose a scene image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        img_resized = img.resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        prediction = model.predict(img_array)
        predicted_class = CLASSES[np.argmax(prediction)]

        st.markdown("---")
        st.markdown(
            f"<h2 style='text-align: center; color: #4CAF50;'>✅ Prediction: <b>{predicted_class.upper()}</b></h2>",
            unsafe_allow_html=True
        )

# ------------- Tab 2: Evaluation Dashboard -------------
with tab2:
    st.header("📊 Model Evaluation Dashboard")

    # Accuracy
    st.subheader("✅ Model Accuracy")
    st.markdown("**Test Accuracy:** ~84%")

    # Confusion matrix
    st.subheader("📉 Confusion Matrix")
    if os.path.exists(CONFUSION_MATRIX_PATH):
        st.image(CONFUSION_MATRIX_PATH, caption="Confusion Matrix")

    # Classification report (summary only)
    st.subheader("📋 Classification Report")
    st.code("""
                  precision    recall  f1-score   support

       buildings       0.82      0.89      0.85
          forest       0.95      0.98      0.97
         glacier       0.83      0.78      0.80
        mountain       0.88      0.61      0.72
             sea       0.71      0.94      0.81
          street       0.90      0.86      0.88

        accuracy                           0.84
       macro avg       0.85      0.84      0.84
    weighted avg       0.85      0.84      0.84
    """)

    st.markdown("---")
    st.caption("Built with ❤️ using TensorFlow + Streamlit")

# -------------------------------------------------------
