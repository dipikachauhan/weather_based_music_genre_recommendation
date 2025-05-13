 🎵 Weather-Based Music Genre Recommender
A deep learning-powered web app that detects the weather condition from an uploaded image and recommends a suitable music genre based on the predicted weather!
🌦 Overview
This project leverages a Convolutional Neural Network (CNN) model based on VGG16 to classify weather conditions from images. It is integrated into a user-friendly 
Streamlit web interface, which then recommends music genres tailored to the detected weather.

🚀 Features
- Weather classification using deep learning (VGG16 with transfer learning)
- Real-time image upload and prediction
- Music genre recommendations tailored to:
  - 🌧 Rain
  - ❄ Snow
  - ☁ Cloudy
  - ⚡ Lightning
  - 🌅 Sunrise
  - 🌈 Rainbow
  - 🌨 Hail

🛠 Tech Stack
- Python
- TensorFlow/Keras (for model training)
- Streamlit (for web interface)
- OpenCV, PIL, NumPy (for image processing)
- Matplotlib (for visualizing predictions)

## 📁 File Structure
- `weather_based_genre_recommender.py` – Model training, data preparation, prediction logic
- `app.py` – Streamlit web application
- `weather_image_detection.h5` – Trained weather classification model
