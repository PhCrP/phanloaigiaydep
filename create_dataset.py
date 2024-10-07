import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder

# image_dir_submain = 'content/submain/Shoe_Sandal_Boot'
image_dir_main = 'content/submain/Shoe_Sandal_Boot'
categories = ['Boot', 'Sandal', 'Shoe']

label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(categories)


def load_images_from_folders(image_dir, categories, img_size=(256, 256)):
    images = []
    labels = []
    for label, category in enumerate(categories):
        folder_path = os.path.join(image_dir, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                images.append(img_resized)
                labels.append(label)
    return np.array(images), np.array(labels)


X, y = load_images_from_folders(image_dir_main, categories)


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


X_hog = extract_hog_features(X)
# X_hog_sub = extract_hog_features(X_sub)
X_sift = extract_sift_features(X)
# X_sift_sub = extract_sift_features(X_sub)

X_combined = np.hstack((X_hog, X_sift))

df = pd.DataFrame(X_combined)
df['label'] = y

# df.to_csv('data/submain/data.csv', index=False)
df.to_csv('data/submain/data.csv', index=False)
print("Thành công")
