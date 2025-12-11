"""
Train SVM model untuk cat & dog classification dan simpan model
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def load_images_from_folder(folder_path):
    """Load all images from folder structure: folder/class_name/image.jpg"""
    images = []
    labels = []
    
    class_names = ['cats', 'dogs']
    label_map = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_path = Path(folder_path) / class_name
        if class_path.exists():
            img_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
            for img_file in img_files:
                try:
                    images.append(str(img_file))
                    labels.append(label_map[class_name])
                except:
                    pass
    
    return images, labels

def create_features(path):
    img = Image.open(path)
    # Resize using PIL directly (faster)
    img = img.resize((56, 56), Image.Resampling.LANCZOS)
    img_arr = np.array(img)
    
    # Handle grayscale images
    if len(img_arr.shape) == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    
    # flatten three channel color image
    color_features = img_arr.flatten()
    # convert image to greyscale
    grey_image = rgb2gray(img_arr)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(8, 8))
    # combine color and hog features into a single array
    flat_features = np.hstack((color_features, hog_features))
    return flat_features

def create_feature_matrix(img_paths):
    features_list = []
    for img_path in img_paths:
        img_features = create_features(img_path)
        features_list.append(img_features)
    return np.array(features_list)

if __name__ == "__main__":
    print("Loading images...")
    data_dir = Path.cwd() / "cats and dogs"
    imgs, labels = load_images_from_folder(data_dir)
    print(f"Loaded {len(imgs)} images")
    
    print("Creating features...")
    feature_matrix = create_feature_matrix(imgs)
    print(f"Feature matrix shape: {feature_matrix.shape}")
    
    print("Scaling features...")
    scaler = StandardScaler()
    imgs_stand = scaler.fit_transform(feature_matrix)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        imgs_stand, labels, test_size=0.3, random_state=1234123
    )
    
    print("Training SVM model...")
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    print("Evaluating model...")
    train_score = svm.score(X_train, y_train)
    test_score = svm.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    print("Saving model and scaler...")
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully!")
