"""
Streamlit app untuk Cat & Dog Classification
"""
import streamlit as st
import pickle
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from pathlib import Path

# Lokasi file model relatif ke app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# Load model dan scaler
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model atau scaler tidak ditemukan")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def create_features(img):
    """Extract features dari gambar"""
    # Resize gambar using PIL directly
    img = img.resize((56, 56), Image.Resampling.LANCZOS)
    
    # Convert to RGB if RGBA or grayscale
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_arr = np.array(img)
    
    # Flatten three channel color image
    color_features = img_arr.flatten()
    
    # Convert image to greyscale
    grey_image = rgb2gray(img_arr)
    
    # Get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(8, 8))
    
    # Combine color and hog features into a single array
    flat_features = np.hstack((color_features, hog_features))
    return flat_features

def predict_image(img, model, scaler):
    """Predict apakah gambar adalah cat atau dog"""
    # Extract features
    features = create_features(img)
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability

# Streamlit UI
st.title("🐱 Cat & Dog Classification 🐶")
st.write("Upload gambar untuk memprediksi apakah itu kucing atau anjing")

# Load model
try:
    model, scaler = load_model()
    st.success(f"Model loaded: {MODEL_PATH.name} | Scaler: {SCALER_PATH.name}")
except FileNotFoundError:
    st.error("Model tidak ditemukan! Jalankan train_model.py atau sel simpan model di notebook.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar yang diupload:")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Hasil Prediksi:")
        
        # Predict
        with st.spinner('Memprediksi...'):
            prediction, probability = predict_image(image, model, scaler)
        
        # Display result
        label_map = {0: '🐱 Cat', 1: '🐶 Dog'}
        predicted_label = label_map[prediction]
        
        st.markdown(f"### {predicted_label}")
        
        # Display probabilities
        st.write("**Confidence:**")
        st.progress(float(probability[prediction]))
        st.write(f"Cat: {probability[0]*100:.2f}%")
        st.write(f"Dog: {probability[1]*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("*Model: SVM with Linear Kernel*")
