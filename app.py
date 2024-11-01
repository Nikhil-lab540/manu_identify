from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
from pdf2image import convert_from_path
import tempfile
import shutil
import base64
import requests

from langchain_openai import ChatOpenAI

st.set_page_config("IMAGE ANALYSIS")
uploaded_file = st.file_uploader(label="Upload your file here ")

# GEMINI 1.5 FLASH
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# # OpenAI API Key
# api_key = os.getenv("OPENAI_API_KEY")


# Function to analyze images with GPT-4o-mini
# def analyze_images_with_gpt(encoded_images, prompt):
#     st.write("Processing with GPT-4o-mini")
#     answer = ""
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
    # }
    # for base64_image in encoded_images:
    #     payload = {
    #         "model": "gpt-4o-mini",
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": prompt
    #                     },
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {
    #                             "url": f"data:image/jpeg;base64,{base64_image}"
    #                         }
    #                     }
    #                 ]
    #             }
    #         ],
    #         "max_tokens": 75
    #     }

    #     response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    #     response_json = response.json()
    #     content = response_json['choices'][0]['message']['content']
    #     answer += content
    # return answer

# Function to process images with GEMINI
def gemini(images, prompt):
    
    answer = ""
    for image in images:
        with open(image, "rb") as img_file:
            myfile = genai.upload_file(image)

        result = gemini_model.generate_content([myfile, "\n\n", prompt])
        answer += result.text
    return answer

# Function to analyze dimensions with GEMINI
def analyze_dim(image, prompt):
    
    myfile = genai.upload_file(image)
    result = gemini_model.generate_content([myfile, "\n\n", prompt])
    return result.text



# # Function to encode the image
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# Creating a temporary path for gemini input and encode images f0r the gpt json input 
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        shutil.copyfileobj(uploaded_file, tmpfile)
        file_path = tmpfile.name
    st.write("File uploaded...")

    # Path to your PDF file
    pdf_path = file_path

    # Convert PDF to images (each page will be converted to an image)
    pages = convert_from_path(pdf_path, 100)  # 300 is the resolution (DPI)

    # Save each page as a separate image
    images = []
    # encoded_images = []
    progress_bar = st.progress(0)  # Initialize progress bar
    total_pages = len(pages)
    for i, page in enumerate(pages):
        image_path = f'output_image_page_{i + 1}.png'
        page.save(image_path, 'PNG')
        progress = (i + 1) / total_pages
        progress_bar.progress(progress)  # Update progress bar
    
        # # Encode the image
        # encoded_image = encode_image(image_path)
        # encoded_images.append(encoded_image)
        
        images.append(image_path)
        

    # Button prompts
    if st.button("Describe the images"):
        with st.spinner("Processing images..."):
            prompt = "Can you describe the images?"
            response = gemini(images, prompt)
            st.write(response)

    if st.button("Analyze the dimensions"):
        with st.spinner("Processing images..."):
            prompt = "Can you analyze the photo and give me the outer dimensions such as and give length , bredth and height  in only integers not float for the component?"
            response = analyze_dim(images[0], prompt)
            st.write(response)

    if st.button("Capture text and translate"):
        with st.spinner("Processing images..."):
            prompt = "You are a manufacturing engineer expert. These are the blueprints for the components that need to be manufactured. There are notes written either in German or English in the image somewhere. Carefully analyze the images and  translate them to English without losing any content and do not give any German content."
            response = gemini(images, prompt)
            st.write(response)

    if st.button("Provide manufacturing process"):
        with st.spinner("Processing images..."):
            prompt = "You are a manufacturing engineer expert. These are the blueprints for the components that need to be manufactured. Carefully analyze the photos and provide a manufacturing process.For example if there is a hole you should say punching operation is used, or if it is pressed say pressing operation is used and also provide the dimensions used by studying the blueprints .DO NOT GIVE WRONG INSTRUCTIONS."
            response = gemini(images, prompt)
            st.write(response)

    if st.button("Verify the edges"):
        with st.spinner("Processing images..."):
            prompt = "Analyze each picture and take any reference picture and verify whether the internal and external dimensions are correct."
            response = gemini(images, prompt)
            st.write(response)

    # if st.button("Describe images with GPT-4o-mini"):
    #     prompt = "Can you describe the images?"
    #     response = analyze_images_with_gpt(encoded_images, prompt)
    #     st.write(response)

    # if st.button("Capture text and translate using GPT-4o-mini"):
    #     prompt = "You are a manufacturing engineer expert. These are the blueprints for the components that need to be manufactured. There are notes written either in German or English in the image somewhere. Carefully analyze the images and find the notes and translate them to English without losing any content and do not give any German content."
    #     response = analyze_images_with_gpt(encoded_images, prompt)
    #     st.write(response)
    query = st.text_area("Enter queries")
    if query:
        prompt ="Answer the questions of the user"
        response= gemini(images,prompt)
        st.write(response)