import streamlit as st
import cv2
from keras.models import load_model
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = [name.strip() for name in open("labels.txt", "r").readlines()]

def classify_frame(frame):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # resizing the frame to be at least 224x224
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # turn the frame into a numpy array
    frame_array = np.asarray(frame)

    # Normalize the frame
    normalized_frame_array = (frame_array.astype(np.float32) / 127.5) - 1

    # Load the frame into the array
    data[0] = normalized_frame_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class Name:", class_name)  # Add this print statement for debugging purposes

    return class_name, confidence_score

def main():
    st.title("")

    cap = cv2.VideoCapture(0)  # Use the default webcam (you can change the index if you have multiple webcams)

    # Create placeholders for displaying live camera feed and predictions
    camera_placeholder = st.empty()
    animal = st.empty()
    prediction_placeholder = st.empty()


    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            class_name, confidence_score = classify_frame(frame)

            # Display the live camera feed with OpenCV
            camera_placeholder.image(frame, channels="BGR", use_column_width=True, caption="Live Webcam Feed")
            if confidence_score >= 0.75:
                animal.write(f"Class: {class_name}")

                # Display only one answer at a time
                prediction_placeholder.subheader("Real-time Prediction:")
                prediction_placeholder.write(f"Confidence Score: {confidence_score}")

if __name__ == "__main__":
    main()
