import streamlit as st
import pickle
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from pathlib import Path

# --- KONFIGURASI HALAMAN (Wajib di baris pertama) ---
st.set_page_config(
    page_title="Pet Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Aesthetic Blue & White) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1E40AF; /* Dark Blue */
        font-weight: 700;
    }
    
    /* Container styling (Card Effect) */
    .css-1r6slb0, .stFileUploader {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
    }

    /* Custom Button */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }

    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    
    /* Result Box Highlight */
    .result-box {
        background-color: #EFF6FF; /* Light Blue */
        border-left: 5px solid #3B82F6;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOGIC SETUP ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model files not found")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def create_features(img):
    img = img.resize((56, 56), Image.Resampling.LANCZOS)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_arr = np.array(img)
    color_features = img_arr.flatten()
    grey_image = rgb2gray(img_arr)
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(8, 8))
    flat_features = np.hstack((color_features, hog_features))
    return flat_features

def predict_image(img, model, scaler):
    features = create_features(img)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    return prediction, probability

# --- SIDEBAR (Info & Status) ---
with st.sidebar:
    st.title("AI Detector")
    st.info("Aplikasi ini menggunakan Support Vector Machine (SVM) dengan ekstraksi fitur HOG & Color Histogram.")
    
    st.markdown("---")
    # Load model status indicator
    try:
        model, scaler = load_model()
        st.success(f"✅ Model Siap")
        st.caption(f"Source: {MODEL_PATH.name}")
    except FileNotFoundError:
        st.error("❌ Model Missing")
        st.stop()
    
    st.markdown("---")
    st.markdown("Created with ❤️ by **You**")

# --- MAIN CONTENT ---
# Header Section
st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>🐾 Cat or Dog?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B;'>Upload foto hewan peliharaanmu, biar AI yang menebak!</p>", unsafe_allow_html=True)

st.write("") # Spacer

# File Uploader Section
uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Processing layout
    st.markdown("---")
    
    image = Image.open(uploaded_file)
    
    # Layout 2 Columns
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown("#### Input Image")
        # Menampilkan gambar dengan border radius css effect (via st.image wrapper)
        st.image(image, use_container_width=True, caption="Uploaded Photo")
    
    with col2:
        st.markdown("#### Analysis Result")
        
        with st.spinner('Sedang menganalisis pixel...'):
            prediction, probability = predict_image(image, model, scaler)
        
        # Mapping Label
        label_map = {0: 'Cat', 1: 'Dog'}
        icon_map = {0: '🐱', 1: '🐶'}
        
        predicted_label = label_map[prediction]
        predicted_icon = icon_map[prediction]
        confidence = probability[prediction]
        
        # Display Result Card
        st.markdown(f"""
        <div class="result-box">
            <h2 style="margin:0; color:#1E3A8A;">{predicted_icon} {predicted_label}</h2>
            <p style="margin:0; color:#64748B;">Confidence Score: <b>{confidence*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer
        
        # Detailed Probabilities
        st.caption("Probability Breakdown:")
        
        # Cat Bar
        st.write(f"🐱 Cat: **{probability[0]*100:.1f}%**")
        st.progress(float(probability[0]))
        
        # Dog Bar
        st.write(f"🐶 Dog: **{probability[1]*100:.1f}%**")
        st.progress(float(probability[1]))

else:
    # Empty State (Tampilan saat belum upload)
    st.markdown("""
    <div style='text-align: center; padding: 50px; color: #94A3B8; border: 2px dashed #CBD5E1; border-radius: 10px;'>
        <h3>🖼️ Belum ada gambar</h3>
        <p>Silakan upload file gambar di atas untuk memulai prediksi.</p>
    </div>
    """, unsafe_allow_html=True)