import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import Image
import os

st.title("N17 Phân loại giày dép")
st.title("(Boot, Sandal, Shoe)")

categories = ['Boot', 'Sandal', 'Shoe']
label_encoder = LabelEncoder()
label_encoder.fit(categories)


def extract_hog_features(images):
    hog_features = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(
            img_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True
        )
        hog_features.append(features)
    return np.array(hog_features)


def extract_sift_features(images):
    sift = cv2.SIFT_create()
    sift_features = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)
        if descriptors is not None:
            mean_descriptor = np.mean(descriptors, axis=0)
            sift_features.append(mean_descriptor)
        else:
            sift_features.append(np.zeros(128))
    return np.array(sift_features)


@st.cache_resource
def load_model(model_type):
    
    if model_type == "SVM":
        model_path = 'data_train/svm/model_svm.sav'
        scaler_path = 'data_train/svm/scaler_svm.sav'
    elif model_type == "CART":
        model_path = 'data_train/cart/model_cart.sav'
        scaler_path = 'data_train/cart/scaler_cart.sav'
    elif model_type == "Neural Network":
        model_path = 'data_train/mlp/model_mlp.sav'
        scaler_path = 'data_train/mlp/scaler_mlp.sav'
    elif model_type == "Bagging": 
        model_paths = {
            'svm': 'data_train/bag/bagging_svm_model.sav',
            'cart': 'data_train/bag/bagging_cart_model.sav'
        }
        scaler_path = 'data_train/bag/scaler_model.sav'
    else:
        st.error("Mô hình đang phát triển")
        return None, None

    if model_type == "Bagging":
        for key, path in model_paths.items():
            if not os.path.exists(path):
                st.error(f"Mô hình {key} đang phát triển")
                return None, None
    else:
        if not os.path.exists(model_path):
            st.error("Mô hình đang phát triển")
            return None, None
        if not os.path.exists(scaler_path):
            st.error("Mô hình đang phát triển")
            return None, None

    if model_type == "Bagging":
        models = {}
        for key, path in model_paths.items():
            with open(path, 'rb') as f:
                models[key] = pickle.load(f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return (models if model_type == "Bagging" else model), scaler


st.sidebar.header("Chọn mô hình")
model_options = ["SVM", "CART", "Neural Network", "Bagging"]
selected_model = st.sidebar.selectbox(
    "Chọn mô hình huấn luyện:", model_options)

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (256, 256))

        st.image(image, caption="Upload ok", use_column_width=True)

        hog_features = extract_hog_features([img_resized])
        sift_features = extract_sift_features([img_resized])
        combined_features = np.hstack((hog_features, sift_features))

        model, scaler = load_model(selected_model)
        if model is None or scaler is None:
            st.stop()

        combined_features_scaled = scaler.transform(combined_features)

        if selected_model == "Bagging":
            with st.spinner('Đang dự đoán...'):
                y_pred_svm = model['svm'].predict(combined_features_scaled)
                y_pred_cart = model['cart'].predict(combined_features_scaled)

                y_pred_combined = (y_pred_svm + y_pred_cart) / 2

                # Round to get final class
                y_pred_combined_rounded = np.round(y_pred_combined).astype(int)
                predicted_label = label_encoder.inverse_transform([y_pred_combined_rounded[0]])[0]
        else:
            with st.spinner('Đang dự đoán...'):
                prediction = model.predict(combined_features_scaled)
                predicted_label = label_encoder.inverse_transform([prediction[0]])[0]

        if predicted_label in categories:
            st.success(f"Phân loại: **{predicted_label}**")
        else:
            st.warning("Không phải giày dép.")

    except Exception as e:
        st.error(f"Thử lại: {e}")
