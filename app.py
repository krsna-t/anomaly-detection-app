import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

# Load TFLite model and allocate tensors
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

def load_labels(path="labels.txt"):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image, interpreter, labels):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_array = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output)
    return labels[predicted_index], output[0][predicted_index], output[0]

def show_prediction_bar(labels, predictions):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.bar(labels, predictions, color=['green', 'red'])
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    plt.title("Prediction Confidence")
    st.pyplot(plt)

# Streamlit UI
st.title("Anomaly Detection ")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Detecting..."):
        interpreter = load_model()
        labels = load_labels()
        result, confidence, full_preds = predict(image, interpreter, labels)
        time.sleep(1)
        st.success(f"Prediction: **{result}** with confidence {confidence:.2f}")
        show_prediction_bar(labels, full_preds)
