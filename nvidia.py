import streamlit as st
import os
import google.generativeai as genai
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pdf2image import convert_from_path
import tempfile
import shutil
import requests
import base64

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure API Keys
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["NVIDIA_API_KEY"] = os.getenv("Nvidia")

# Load models
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
nvidia_llm = ChatNVIDIA(model="nvidia/neva-22b")

# Set up Streamlit page
st.set_page_config("PDF to Image Analysis")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload your PDF file here", type=['pdf'])

def process_pdf(pdf_file):
    """Converts PDF to images."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        shutil.copyfileobj(pdf_file, tmpfile)
        pdf_path = tmpfile.name

    pages = convert_from_path(pdf_path, 300)  # 300 DPI for high resolution
    images = []
    total_pages = len(pages)
    progress_bar = st.progress(0)
    
    for i, page in enumerate(pages):
        image_path = f'output_image_page_{i + 1}.png'
        page.save(image_path, 'PNG')
        images.append(image_path)
        progress_bar.progress((i + 1) / total_pages)
        
    return images

def analyze_images_gemini(images):
    """Analyze images using the Gemini 1.5 Flash model."""
    results = []
    for image in images:
        myfile = genai.upload_file(image)
        prompt = (
            "You are a technical expert analyzing a blueprint. "
            "Please provide the following details: "
            "1. Dimensions (length, breadth, height) "
            "2. Part number "
            "3. Material "
            "4. Step-by-step manufacturing process"
        )
        result = gemini_model.generate_content([myfile, "\n\n", prompt])
        results.append(result.text)
    return results

def analyze_images_nvidia(images):
    """Use Nvidia Neva-22b for analyzing the images directly."""
    results =[]

    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
    stream = False
    prompt = (
            "You are a technical expert analyzing a blueprint. "
            "Please provide the following details: "
            "1. Dimensions (length, breadth, height) "
            "2. Part number "
            "3. Material "
            "4. Step-by-step manufacturing process"
        )
    for image in images:
        with open(image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()


        headers = {
        "Authorization": "Bearer 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",
        "Accept": "text/event-stream" if stream else "application/json"
        }

        payload = {
        "messages": [
            {
            "role": "user",
            "content": f'{prompt}. <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        
        "temperature": 1,

        }

        response = requests.post(invoke_url, headers=headers, json=payload)


        content = response.json()['choices'][0]['message']['content']
        results.append(content)
    return results

if uploaded_file is not None:
    st.write("Processing your PDF file...")

    # Process PDF and convert to images
    images = process_pdf(uploaded_file)
    st.write(f"{len(images)} images have been extracted from the PDF.")

    # Analyze images with Gemini 1.5 Flash
    if st.button("Analyze Images with Gemini"):
        with st.spinner("Analyzing images..."):
            gemini_analysis = analyze_images_gemini(images)
            for idx, result in enumerate(gemini_analysis):
                st.write(f"**Gemini Analysis for Image {idx + 1}:** {result}")

    # Analyze images with Nvidia Neva-22b
    if st.button("Analyze with Nvidia"):
        with st.spinner("Refining analysis with Nvidia Neva-22b model..."):
            nvidia_analysis = analyze_images_nvidia(images)
            for idx, result in enumerate(nvidia_analysis):
                st.write(f"**Nvidia Analysis for Image {idx + 1}:** {result}")
    
    # Clean up temporary files
    for image in images:
        os.remove(image)
