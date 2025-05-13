import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(page_title="🎵 Weather-Based Music Recommender", layout="centered")
st.title("🌦️ Weather Detection and Music Genre Recommendation")
st.markdown("Upload a weather image to detect the weather condition and get a recommended music genre!")

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("weather_image_detection.h5")
    return model

model = load_model()

# Class labels
weather_classes = ['Cloudy', 'Sunrise', 'hail', 'lightning', 'rain', 'rainbow', 'snow']

# Music recommendations based on weather
music_recommendations = {
    'cloudy': ['Lo-fi Chill ☁️', 'Indie Folk 🍂', 'Ambient Electronic 🎧'],
    'Sunrise': ['Soft Pop 🌅', 'Chillwave 🌇', 'Acoustic Morning 🎶'],
    'hail': ['Heavy Metal 🤘', 'Thrash Metal ⚔️', 'Punk Rock 🎸'],
    'lightning': ['Electronic Dance ⚡', 'Industrial Rock 🔥', 'Dubstep 💥'],
    'rain': ['Acoustic Sad Songs ☔', 'Jazz Blues 🎸', 'Piano Ballads 🎹'],
    'rainbow': ['Happy Indie 🌈', 'Psychedelic Rock ✨', 'Disco Funk 🕺'],
    'snow': ['Classical Piano ❄️', 'Winter Jazz ⛄', 'Orchestral Soundtracks 🎻']
}
# File upload UI
uploaded_file = st.file_uploader("📤 Upload a weather image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    IMG_SIZE = 256
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalization
    img = np.expand_dims(img, axis=0)

    # Make prediction
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions)
    predicted_weather = weather_classes[predicted_index]
    confidence = float(np.max(predictions))

    # Get recommended music
    recommended_music = music_recommendations.get(predicted_weather, "Unknown Genre")

    # Display results
    st.markdown("---")
    st.subheader("🌤️ Weather Detection Result")
    st.success(f"**Weather Condition:** `{predicted_weather}`")
    st.info(f"**Confidence:** `{confidence:.2f}`")

    st.markdown("---")
    st.subheader("🎶 Recommended Music Genre")
    st.success(f"**Genre:** `{recommended_music}`")

    # Optional: Show confidence scores
    st.markdown("### 📊 Confidence Scores for All Weathers")
    for i, label in enumerate(weather_classes):
        st.write(f"{label}: `{predictions[0][i]:.2f}`")
else:
    st.info("Please upload a weather image to detect and get a music recommendation.")
