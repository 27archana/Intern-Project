import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import urllib.request

# Define remote URLs for your files
EMBEDDINGS_URL = "https://github.com/27archana/Intern-Project/releases/download/v1.0.0/embeddings.pkl"
FILENAMES_URL = "https://github.com/27archana/Intern-Project/releases/download/v1.0.0/filenames.pkl"

# Helper function to download files if not present
def load_pickle(name, url):
    if not os.path.exists(name):
        urllib.request.urlretrieve(url, name)
    with open(name, 'rb') as f:
        return pickle.load(f)

# Load embeddings and filenames
feature_list = np.array(load_pickle("embeddings.pkl", EMBEDDINGS_URL))
filenames = load_pickle("filenames.pkl", FILENAMES_URL)

# Load model
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    return model

model = load_model()

# App Title
st.title("👗 Fashion Recommender System")

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except:
        return None

# Feature extraction function
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Recommendation function
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File uploader
uploaded_file = st.file_uploader("📁 Choose an image")

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)
        features = feature_extraction(file_path, model)
        indices = recommend(features, feature_list)

        st.subheader("🛍 You might also like:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            col.image(filenames[indices[0][i]])
    else:
        st.error("⚠️ Error saving the uploaded file.")
