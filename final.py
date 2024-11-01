



import streamlit as st
import os
import google.generativeai as genai
from pdf2image import convert_from_path
import tempfile
import shutil
from google.ai.generativelanguage_v1beta.types import content
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API Keys
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Generation config for text analysis
generation_config_for_text = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    
    "response_schema": content.Schema(
        type=content.Type.OBJECT,
        properties={
            "manufacturing_process": content.Schema(type=content.Type.STRING),
            
            "material": content.Schema(type=content.Type.STRING),
            
            
            "part_number": content.Schema(type=content.Type.STRING),
            "notes": content.Schema(type=content.Type.STRING),
        },
    ),
    "response_mime_type": "application/json",
}

# Generation config for dimension analysis
generation_config_for_dimensions = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type=content.Type.OBJECT,
        properties={
            "length": content.Schema(type=content.Type.NUMBER),
            "breadth": content.Schema(type=content.Type.NUMBER),
            "height": content.Schema(type=content.Type.NUMBER),
        },
    ),
    "response_mime_type": "application/json",
}

# Load models
gemini_model_for_text = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config_for_text)
gemini_model_for_dimensions = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config_for_dimensions)

# Set up Streamlit page
st.set_page_config("PDF to Image Analysis")

uploaded_file = st.file_uploader("Upload your PDF file here")

def process_pdf(pdf_file):
    """Converts PDF to images."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        shutil.copyfileobj(pdf_file, tmpfile)
        pdf_path = tmpfile.name
    pages = convert_from_path(pdf_path, 300)  # 300 DPI for high resolution
    images = []
    for i, page in enumerate(pages):
        image_path = f'output_image_page_{i + 1}.png'
        page.save(image_path, 'PNG')
        images.append(image_path)
    return images

def upload_to_gemini(path, mime_type="image/jpeg"):
    """Uploads the given image file to Gemini."""
    return genai.upload_file(path, mime_type=mime_type)

def analyze_dimensions(images):
    results = []
    
    uploaded_file = upload_to_gemini(images[0])
    prompt = (
        "Please extract the length, breadth, and height of the object from the provided blueprint. "
        "Make sure to analyze properly the height and double check the measurements and ensure consistency."
        "Height might not be more than 60mm (MIGHT)"
        "The measurements are surrounded by brackets. Dont think the bracket () as 1 and , as ."
    )
    result = gemini_model_for_dimensions.generate_content([uploaded_file, prompt])
    results.append(result.text)
    return results

def analyze_text(images):
    results = []
    
    uploaded_file1 = upload_to_gemini(images[0])
    uploaded_file2 = upload_to_gemini(images[1])
    prompt = (
        "Please extract the part number, material used , and step-by-step manufacturing process from the blueprint. "
        "Translate any text in German to English."
    )
    result = gemini_model_for_text.generate_content([uploaded_file1,uploaded_file2, prompt])
    results.append(result.text)
    return results

if uploaded_file is not None:
    st.write("Processing your PDF file...")

    # Convert PDF to images
    images = process_pdf(uploaded_file)
    st.write(f"{len(images)} images extracted from the PDF.")
    
    with st.spinner("Analyzing dimensions..."):
        dimension_results = analyze_dimensions(images)
    
    with st.spinner("Analyzing textual information..."):
        text_results = analyze_text(images)

    st.write("Analysis Complete!")
    
    # Display results
    for i, result in enumerate(dimension_results):
        st.write(f"Dimensions for Image :")
        st.json(result)
    
    for i, result in enumerate(text_results):
        st.write(f"Textual Analysis for Image :")
        st.json(result)

    #     # Clean up temporary images
    # for image in images:
    #     os.remove(image)