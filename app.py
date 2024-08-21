import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from matplotlib import colors as mcolors
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix



# ColorClassifier class and functions
class ColorClassifier:
    def __init__(self, dataset_path, bins=(8, 8, 8), patch_size=50):
        self.dataset_path = dataset_path
        self.bins = bins
        self.patch_size = patch_size
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.le = LabelEncoder()
        self.color_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
        self.features = []
        self.labels = []



    def extract_color_histogram(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()















    def load_data(self):
        for color in self.color_folders:
            color_folder_path = os.path.join(self.dataset_path, color)
            for image_name in os.listdir(color_folder_path):
                image_path = os.path.join(color_folder_path, image_name)
                if os.path.isfile(image_path): 
                    image = cv2.imread(image_path)
                    if image is not None:
                        hist = self.extract_color_histogram(image)
                        self.features.append(hist)
                        self.labels.append(color)
                    else:
                        print(f"Warning: Could not read image {image_path}")
                else:
                    print(f"Warning: {image_path} is not a file")

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.encoded_labels = self.le.fit_transform(self.labels)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.encoded_labels, test_size=0.2, random_state=42)
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test,y_pred))
        print("Classification Report: ")
        print(classification_report(y_test,y_pred))

    
    def predict_color_distribution(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        height, width, _ = image.shape
        color_counts = np.zeros(len(self.le.classes_))

        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    continue
                hist = self.extract_color_histogram(patch)
                color_index = self.clf.predict([hist])[0]
                color_counts[color_index] += 1

        total_patches = np.sum(color_counts)
        color_percentages = (color_counts / total_patches) * 100
        color_distribution = {self.le.inverse_transform([i])[0]: color_percentages[i] for i in range(len(color_percentages))}

        return color_distribution

def get_color_hex(color_name):
    try:
        color_hex = mcolors.CSS4_COLORS[color_name.lower()]
    except KeyError:
        color_hex = '#000000'  # Default to black if color is not found
    return color_hex






def generate_color_table(color_distribution):
    table_html = '<table style="width:100%; border-collapse: collapse;">'
    table_html += '<tr><th>Color</th><th>Percentage</th></tr>'
    
    for color, percentage in color_distribution.items():
        if color.lower() != 'white':
            color_hex = get_color_hex(color)
            table_html += f'<tr><td style="color: {color_hex};">{color}</td><td>{percentage:.2f}%</td></tr>'
        else:
            table_html += f'<tr><td>{color}</td><td>{percentage:.2f}%</td></tr>'
    
    table_html += '</table>'
    return table_html





















def main():
    st.title("Color Classifier")
    st.write("Upload an image and see the color distribution.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        # st.write("Classifying...")

        # Save the uploaded file to disk
        uploaded_file_path = os.path.join("uploads", uploaded_file.name)
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict color distribution
        color_distribution = color_classifier.predict_color_distribution(uploaded_file_path)
        
        # Generate and display the color table
        color_table_html = generate_color_table(color_distribution)
        st.markdown(color_table_html, unsafe_allow_html=True)

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    dataset_path = 'ColorClassification_dataset'
    color_classifier = ColorClassifier(dataset_path)
    color_classifier.load_data()
    color_classifier.train_model()
    main()
