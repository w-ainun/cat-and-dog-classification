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

# Load model dan scaler
@st.cache_resource
def load_model():
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def create_features(img):
    """Extract features dari gambar"""
    # Resize gambar using PIL directly
    img = img.resize((56, 56), Image.Resampling.LANCZOS)
    img_arr = np.array(img)
    
    # Handle grayscale images
    if len(img_arr.shape) == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    
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
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model tidak ditemukan! Jalankan train_model.py terlebih dahulu.")
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
