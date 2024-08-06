import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

st.title("Image Captioning")

# URL input
url = st.text_input("Enter image URL (optional):")
if url:
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        st.image(image, caption="Image from URL", use_column_width=True)
        caption = generate_caption(image)
        st.write(f"Caption: {caption}")
    except Exception as e:
        st.error(f"Error fetching image from URL: {e}")

# File upload
uploaded_file = st.file_uploader("Upload an image file (optional):", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    caption = generate_caption(image)
    st.write(f"Caption: {caption}")

if not url and not uploaded_file:
    st.write("Please enter an image URL or upload an image file to get a caption.")
