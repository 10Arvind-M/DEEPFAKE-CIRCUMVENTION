import streamlit as st
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.tensorboard import SummaryWriter
import tempfile
import shutil

# Create the directory for uploaded videos if it doesn't exist
if not os.path.exists('uploaded_videos'):
    os.makedirs('uploaded_videos')

# Load the model and image processor
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=2)
model.load_state_dict(torch.load('vit_deepfake_model.pth', map_location=torch.device('cpu')))
model.eval()

image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Initialize TensorBoard
writer = SummaryWriter(log_dir='logs')

def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        if len(frames) == num_frames:
            break
    cap.release()
    return frames

def predict_video(video_path):
    frames = extract_frames(video_path)
    inputs = image_processor(images=frames, return_tensors="pt")
    outputs = model(**inputs).logits
    preds = torch.argmax(outputs, dim=1).numpy()
    confidences = torch.softmax(outputs, dim=1).max(dim=1).values.numpy()  # Get confidence scores
    writer.add_histogram('predictions', preds)
    writer.add_histogram('confidence_scores', confidences)
    return preds.mean(), preds, confidences

def plot_confidence_scores(confidences):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(confidences)), confidences, color='blue')
    plt.xlabel('Frame')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Scores for Each Frame')
    plt.xticks(range(len(confidences)), [f"Frame {i+1}" for i in range(len(confidences))])
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('confidence_scores.png')

# Streamlit layout
st.set_page_config(page_title="Deepfake Video Detector", page_icon="🎥", layout="centered")

st.title("Deepfake Video Detector 🎥")
st.write("Upload a video to predict whether it is real or fake. The model analyzes the video and provides a confidence score.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # Display the uploaded video
    st.video(video_path)

    if st.button('Predict'):
        # Predict the uploaded video
        prediction, preds, confidences = predict_video(video_path)
        
        st.markdown("### Prediction Result")
        
        if prediction > 0.5:
            st.error("The video is likely **fake**!")
        else:
            st.success("The video is likely **real**.")
        
        # Confidence Score Table
        st.markdown("### Confidence Score Legend")
        confidence_data = {
            "Frame": [f"Frame {i+1}" for i in range(len(preds))],
            "Prediction": ["Fake" if p == 1 else "Real" for p in preds],
            "Confidence Score": [round(c, 2) for c in confidences]
        }
        st.table(confidence_data)
        
        # Plot and display confidence scores
        plot_confidence_scores(confidences)
        st.image('confidence_scores.png', caption='Confidence Scores for Each Frame')
        
        st.markdown("### Understanding Confidence Scores")
        st.write("""
            - **Confidence Score**: The confidence score indicates the model's certainty in its prediction. 
            - Scores close to 1 indicate a high probability that the frame is fake.
            - Scores close to 0 indicate a high probability that the frame is real.
            - The final prediction is an average of individual frame scores.
        """)

    # Clean up temporary files
    if os.path.exists(video_path):
        os.remove(video_path)
else:
    st.info("Please upload a video to analyze.")

# Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stAlert {
            border: 1px solid #ff4b4b;
            border-radius: 4px;
        }
        .stSuccess {
            border: 1px solid #4bb543;
            border-radius: 4px;
        }
        .stTable {
            border: 1px solid #dddddd;
            border-radius: 4px;
            margin-top: 20px;
        }
        .css-1cpxqw2.e1fqkh3o2 {
            padding-top: 10px;
            padding-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)
