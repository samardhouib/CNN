import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import time

# Charger le modèle
model = load_model('model/model.h5')

# Correspondance des indices de classe avec les noms de classe
indices_to_classes = {
    0: 'Cherry Coke',
    1: 'Fanta',
    2: 'Minute Maid',
    3: 'Nestea',
    4: 'Sprite',
    5: 'aquarius',
    6: 'barrcola',
    7: 'boga',
    8: 'irnbru',
    9: 'j20',
    10: 'prime',
    11: 'rio',
    12: 'rubicon',
    13: 'tango'
}

# Classes à boycotter
boycott_classes = ['Cherry Coke', 'Fanta', 'Minute Maid', 'Nestea', 'Sprite', 'aquarius']

# Fonction pour prétraiter l'image
def preprocess_image(image):
    img = image.resize((299, 299))  # Adapter à la taille d'entrée du modèle (299, 299, 3)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normaliser les valeurs de pixel entre 0 et 1
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension de lot
    return img_array

class ImageClassifier(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.captured_frames = []

    def transform(self, frame):
        self.captured_frames.append(frame)
        return frame

def main():
    st.title('Boycott Israelien  Products')

    # Options pour choisir la source de l'image
    option = st.radio("Choose Image Source", ("Webcam", "Upload Image"))

    if option == "Webcam":
        num_images_to_capture = st.number_input("Enter the number of images to capture", min_value=1, max_value=10, step=1, value=5)

        # Démarrez le flux vidéo de la webcam
        webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=ImageClassifier)

        if webrtc_ctx.video_transformer:
            # Afficher le flux vidéo de la webcam
            st.header('Webcam Feed')
            if len(webrtc_ctx.video_transformer.captured_frames) > 0:
                st.image(webrtc_ctx.video_transformer.captured_frames[-1].to_ndarray(format="bgr24"), use_column_width=True)

            # Ajouter la fonctionnalité de capture d'image
            if st.button("Capture Images"):
                st.write("Please smile! Capturing images...")
                time.sleep(5)  # Attendez 5 secondes
                image_classifier = ImageClassifier()
                for frame in webrtc_ctx.video_transformer.captured_frames:
                    image = Image.fromarray(frame.to_ndarray(format="bgr24"))
                    st.image(image, caption='Captured Image', use_column_width=True)
                    # Ajouter l'image capturée à la liste
                    processed_image = preprocess_image(image)
                    image_classifier.captured_frames.append(processed_image)

                # Prédiction avec le modèle pour les images capturées
                st.write("Predictions for Captured Images:")
                for i, image in enumerate(image_classifier.captured_frames):
                    prediction = model.predict(image)
                    class_index = np.argmax(prediction)
                    class_name = indices_to_classes[class_index]
                    boycott_status = "boycott" if class_name in boycott_classes else "non boycott"
                    st.write(f"Image {i+1}: {class_name}, Boycott Status: {boycott_status}")
                    # Afficher le logo correspondant
                    st.header('Logo')
                    if boycott_status == "boycott":
                        st.image("C:/Users/HP/Desktop/boycott-sign-vector-26247450.webp", width=100)
                    else:
                        st.image("C:/Users/HP/Desktop/safety-logo-vector-48933087.webp", width=100)

    else:
        # Option pour télécharger une image
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Afficher l'image téléchargée
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Prétraitement de l'image
            processed_image = preprocess_image(image)

            # Prédiction avec le modèle
            prediction = model.predict(processed_image)
            class_index = np.argmax(prediction)
            class_name = indices_to_classes[class_index]

            # Vérifier le statut de boycott
            boycott_status = "boycott" if class_name in boycott_classes else "non boycott"
            output = f"Predicted Class: {class_name}, Boycott Status: {boycott_status}"

            # Afficher la prédiction
            st.header('Prediction')
            st.write(output)

            # Afficher le logo correspondant
            st.header('Logo')
            if boycott_status == "boycott":
                st.image("C:/Users/HP/Desktop/boycott-sign-vector-26247450.webp", width=100)
            else:
                st.image("C:/Users/HP/Desktop/safety-logo-vector-48933087.webp", width=100)

if __name__ == "__main__":
    main()
