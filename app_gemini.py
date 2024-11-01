import streamlit as st
import os
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
import tempfile
import shutil
from dotenv import load_dotenv




# Update the path to the location where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Load environment variables
load_dotenv()

# Configure API Key for Gemini 1.5 Flash
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Set up Streamlit page
st.set_page_config("PDF to Image Analysis")

uploaded_file = st.file_uploader("Upload your PDF file here")

def process_pdf(pdf_file):
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        shutil.copyfileobj(pdf_file, tmpfile)
        pdf_path = tmpfile.name

    # Convert PDF to images
    pages = convert_from_path(pdf_path, 300)  # 300 DPI for high resolution
    images = []
    total_pages = len(pages)
    progress_bar = st.progress(0)
    for i, page in enumerate(pages):
        image_path = f'output_image_page_{i + 1}.png'
        page.save(image_path, 'PNG')
        images.append(image_path)
        progress = (i + 1) / total_pages
        progress_bar.progress(progress)
    return images

def ocr_images(images):
    text_results = []
    total_images = len(images)
    progress_bar = st.progress(0)
    
    for i, image in enumerate(images):
        # Perform OCR to extract text from the image
        text = pytesseract.image_to_string(image)
        text_results.append(text)
        
        # Update progress
        progress = (i + 1) / total_images
        progress_bar.progress(progress)
    
    return text_results

def analyze_text(texts):
    results = []
    total_texts = len(texts)
    progress_bar = st.progress(0)

    # Few-shot examples for better accuracy
    few_shot_prompt = """
    Example 1:
    Input: "This technical drawing depicts a metal part with part number 'XYZ123', made of steel, with dimensions of 100mm x 50mm x 10mm. The part includes features such as holes and bends. Manufacturing processes involved are laser cutting and bending."
    Output: Part Number: XYZ123, Material: Steel, Dimensions: 100mm x 50mm x 10mm, Manufacturing Processes: Laser cutting, bending.

    Example 2:
    Input: "This technical drawing depicts a plastic component with part number 'ABC456', made of plastic, with dimensions of 200mm x 100mm x 20mm. It includes features such as slots and grooves. Manufacturing processes involved are injection molding and machining."
    Output: Part Number: ABC456, Material: Plastic, Dimensions: 200mm x 100mm x 20mm, Manufacturing Processes: Injection molding, machining.
    """

    for i, text in enumerate(texts):
        # Prompt with few-shot examples
        prompt = (
            f"{few_shot_prompt}\n\n"
            "Analyze the following text and provide similar detailed information, including part number, material, dimensions, and manufacturing processes."
        )
        
        # Generate content with Gemini
        result = gemini_model.generate_content([text, "\n\n", prompt])
        results.append(result.text)

        # Update progress
        progress = (i + 1) / total_texts
        progress_bar.progress(progress)
    
    return results

if uploaded_file is not None:
    st.write("Processing your PDF file...")

    # Process PDF and convert to images
    images = process_pdf(uploaded_file)
    st.write(f"{len(images)} images have been extracted from the PDF.")

    # Extract text from images using OCR
    if st.button("Extract Text from Images"):
        with st.spinner("Extracting text from images..."):
            ocr_texts = ocr_images(images)
            st.write("Text extraction completed.")
            st.write(ocr_texts)

            # Analyze extracted text with Gemini 1.5 Flash
            # if st.button("Analyze Extracted Text"):
        with st.spinner("Analyzing text..."):
            analysis_results = analyze_text(ocr_texts)
            for idx, result in enumerate(analysis_results):
                st.write(f"### Analysis of Image {idx + 1}")
                st.write(result)

    # Clean up temporary files
    for image in images:
        os.remove(image)
