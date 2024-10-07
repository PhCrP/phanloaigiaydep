import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import Image


def svm_dt():
    model_filename = 'data_train/svm/model_svm.sav'
    svm_model = pickle.load(open(model_filename, 'rb'))

    scaler_filename = 'data_train/svm/scaler_svm.sav'
    sc = pickle.load(open(scaler_filename, 'rb'))

    return svm_model, sc


def cart_dt():
    model_filename = 'data_train/cart/model_cart.sav'
    cart_model = pickle.load(open(model_filename, 'rb'))

    scaler_filename = 'data_train/cart/scaler_cart.sav'
    sc = pickle.load(open(scaler_filename, 'rb'))

    return cart_model, sc


def mlp_dt():
    model_filename = 'data_train/mlp/model_mlp.sav'
    mlp_model = pickle.load(open(model_filename, 'rb'))

    scaler_filename = 'data_train/mlp/scaler_mlp.sav'
    sc = pickle.load(open(scaler_filename, 'rb'))

    return mlp_model, sc


def bag_dt():
    model_filename = 'data_train/bag/model_bag.sav'
    bag_model = pickle.load(open(model_filename, 'rb'))

    scaler_filename = 'data_train/bag/scaler_bag.sav'
    sc = pickle.load(open(scaler_filename, 'rb'))

    return bag_model, sc


categories = ['Boot', 'Sandal', 'Shoe']
label_encoder = LabelEncoder()
label_encoder.fit(categories)


def extract_hog_features(images):
    hog_features = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, _ = hog(img_gray, pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2), visualize=True)
        hog_features.append(features)
    return np.array(hog_features)


def extract_sift_features(images):
    sift = cv2.SIFT_create()
    sift_features = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)
        if descriptors is not None:
            sift_features.append(np.mean(descriptors, axis=0))
        else:
            sift_features.append(np.zeros(128))
    return np.array(sift_features)


st.title("Phân loại giày dép")

model_options = ["SVM", "CART", "Neural Network", "Bagging"]
selected_model = st.selectbox("Chọn mô hình huấn luyện:", model_options)

uploaded_file = st.file_uploader(
    "Chọn ảnh.......", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (256, 256))

    st.image(image, caption="Uploaded Image", use_column_width=True)

    hog_features = extract_hog_features([img_resized])
    sift_features = extract_sift_features([img_resized])
    combined_features = np.hstack((hog_features, sift_features))

    if selected_model == "SVM":
        model, sc = svm_dt()
    elif selected_model == "CART":
        model, sc = cart_dt()
    elif selected_model == "Neural Network":
        model, sc = mlp_dt()
    elif selected_model == "Bagging":
        model, sc = bag_dt()
    else:
        st.error("Invalid model type selected.")

    combined_features = sc.transform(combined_features)

    prediction = model.predict(combined_features)

    predicted_label = label_encoder.inverse_transform([prediction[0]])[0]
    st.write(f"Predicted category: {predicted_label}")
