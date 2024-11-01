# import streamlit as st
# import os
# import google.generativeai as genai
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from pdf2image import convert_from_path
# import tempfile
# import shutil
# import requests
# from google.ai.generativelanguage_v1beta.types import content


# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Configure API Keys
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# os.environ["NVIDIA_API_KEY"] = os.getenv("Nvidia")




# generation_config = {
#     "temperature": 0.1,
#     "top_p": 0.95,
#     "top_k": 64,
#     "max_output_tokens": 8192,
#     "response_schema": content.Schema(
#         type=content.Type.OBJECT,
#         properties={
#             "manufacturing_process": content.Schema(type=content.Type.STRING),
#             "length": content.Schema(type=content.Type.NUMBER),
#             "material": content.Schema(type=content.Type.STRING),
#             "breadth": content.Schema(type=content.Type.NUMBER),
#             "height": content.Schema(type=content.Type.NUMBER),
#             "part_number":content.Schema(type=content.Type.STRING),
#             "notes":content.Schema(type=content.Type.STRING),
#         },
#     ),
#     "response_mime_type": "application/json",
# }

# # Load models
# gemini_model = genai.GenerativeModel("gemini-1.5-flash",
#                                      generation_config=generation_config,
#                                      system_instruction=(
#         "You are a highly skilled technical expert specializing in blueprint analysis. "
#         "Your task is to carefully examine the provided blueprint images and extract detailed information. "
#         "Please provide the following data points: \n"
#         "1. **Dimensions**: Specify the length, breadth, and height of the part in millimeters. \n"
#         "2. **Part Number**: Identify and list the part number associated with the blueprint. \n"
#         "3. **Material**: Describe the material used for the part, including any specifications if available. \n"
#         "4. **Manufacturing Process**: Outline the step-by-step manufacturing process involved in creating this part, detailing any critical steps. \n"
#         "5. **Notes**: identify the text under the heading notes. If it is in German translate it to English and give it in well formated manner. \n"
#         "Ensure that your response is clear, structured, and includes all requested details. If any information is not visible, please indicate that."
#     ),)
# # nvidia_llm = ChatNVIDIA(model="nvidia/neva-22b")



# # Set up Streamlit page
# st.set_page_config("PDF to Image Analysis")

# uploaded_file = st.file_uploader("Upload your PDF file here")

# def process_pdf(pdf_file):
#     # Save the uploaded PDF to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
#         shutil.copyfileobj(pdf_file, tmpfile)
#         pdf_path = tmpfile.name

#     # Convert PDF to images
#     pages = convert_from_path(pdf_path, 300) # 300 DPI for high resolution
#     images = []
#     total_pages = len(pages)
#     progress_bar = st.progress(0)
#     for i, page in enumerate(pages):
#         image_path = f'output_image_page_{i + 1}.png'
#         page.save(image_path, 'PNG')
#         images.append(image_path)
#         progress = (i + 1) / total_pages
#         progress_bar.progress(progress)
#     return images




# # def analyze_images_nvidia(images):
# #     """Use Nvidia Neva-22b for further analysis on extracted text data."""
# #     results = []
# #     # for image in images:
# #     #     prompt = (
# #     #         "Analyze the technical description and extract details about: "
# #     #         "1. Part dimensions "
# #     #         "2. Part number "
# #     #         "3. Material "
# #     #         "4. Suggested manufacturing processes"
# #     #     )
# #     #     result = nvidia_llm.generate({"prompt": prompt})
# #     #     results.append(result["text"])
# #     api_key =os.getenv("Nvidia")

# #     for image in images:
# #         with open(image, "rb") as img_file:
# #             files = {
# #                 'file': img_file
# #             }

# #             headers ={
# #                 "Authorization" : f"Bearer {api_key}",
# #                 "Accept" : 'application/json'
# #             }

# #             payload = {
# #                 "messages": [
# #                     {
# #                         "role": "user",
# #                         "content": f"Analyze the technical description and extract details about: 1. Part dimensions 2. Part number 3. Material 4. Suggested manufacturing processes"
# #                     }
# #                 ],
            
            

# #             }
# #             response = requests.post('NVIDIA_IMAGE_API_URL', headers=headers, json=payload, files=files)  # Add your actual Nvidia image endpoint here.
            
# #             if response.status_code == 200:
# #                 results.append(response.json().get('choices')[0].get('message').get('content'))
# #             else:
# #                 results.append(f"Error: {response.status_code} - {response.text}")

# #     return results
        


# def upload_to_gemini(path, mime_type=None):
#     """Uploads the given file to Gemini."""
#     file = genai.upload_file(path, mime_type=mime_type)
#     return file


# def analyze_images_gemini(images):
#     results = []



#     uploaded_file1 = upload_to_gemini(images[0], mime_type="image/jpeg")
#     uploaded_file2 = upload_to_gemini(images[1], mime_type="image/jpeg")
            

#             # Generate content with the prompt
#             # prompt = ("This is a technical drawing of a metal part. Please provide a detailed analysis which include length breadh height and also material and part numberand provide a detailed and concise manufacturing process in step by step.For example if there is a hole you should say punching operation is used, or if it is pressed say pressing operation is used if possible give the dimensions used for those."
#             #             "Are the inner and outer dimensions matching??"
#             #             "Please explain the notes in English"
#             # )
#     prompt = (
#         "You are a technical expert in an manufacturing factory. you will be provided  images of blueprints of a part."
#         "The provided image is a technical drawing of a metal part. It shows the part from different angles, including a top view, a side view, and a front view."
#         "Please provide a detailed analysis using the following steps."
#         "Here is a technical information."
#         "Please extract the length, breadth, and height of the object from the provided blueprint. Make sure to specify which view (top, front, side) you are extracting these from, and ensure consistency across the views."
#         "Now Identify part number and material used"
#         "Give a detailed step by step process how the part is manufactured. For example if there is a hole you should say punching operation is used, or if it is pressed say pressing operation is used if possible give the dimensions used for those."
#     )
#             # prompt = ("Analyze the following technical drawing and provide a detailed description including part number, material, dimensions, and manufacturing processes.\n\n"
#             #           "Example 1:\nInput: (Image of a technical drawing)\nOutput: This technical drawing depicts a metal part with part number 'XYZ123', made of steel, with dimensions of 100mm x 50mm x 10mm. The part includes features such as holes and bends. Manufacturing processes involved are laser cutting and bending.\n\n"
#             #           "Example 2:\nInput: (Image of another technical drawing)\nOutput: This technical drawing depicts a plastic component with part number 'ABC456', made of plastic, with dimensions of 200mm x 100mm x 20mm. It includes features such as slots and grooves. Manufacturing processes involved are injection molding and machining.\n\n"
#             #           "Analyze the following image and provide similar detailed information:")
            
#             # prompt =("Please provide a detailed analysis of the attached technical drawing of a metal part, following these steps:"

#             #     "Dimensions Analysis: Measure and report the length, breadth, and height of the part."
#             #     "Identification:"
#             #     "Identify the part number."
#             #     "Determine the material used in the part."
#             #     "Manufacturing Process: Outline a step-by-step process of how the part is manufactured."
#             #     "For example, specify if a punching operation is used for creating holes, or a pressing operation for shaping."
#             #     "Include relevant dimensions where applicable (e.g., hole diameter, press depth)."
#             # )
#             # prompt = ("Analyze this technical drawing of a metal part and provide detailed insights, "
#             # "including part number, material, dimensions, and manufacturing processes. "
#             # "If material is not specified take a guess by analyising the image"
#             # "Include any notes or tolerances mentioned in the drawing.")
#     result = gemini_model.generate_content([uploaded_file1,uploaded_file2, "\n\n", prompt])
#     results.append(result.text)
#     # llama = ChatGroq(
#     #     model="llama-3.1-70b-versatile",
#     #     temperature=0,)
    
#     # prompt ="Take this text {result} and simplfiy it"
#     # chain =llama | prompt
#     # text =chain.invoke(results)
    

#     return results



# # if uploaded_file is not None:
# #     st.write("Processing your PDF file...")

# #     # Process PDF and convert to images
# #     images = process_pdf(uploaded_file)
# #     st.write(f"{len(images)} images have been extracted from the PDF.")

# #     # Analyze images with Gemini 1.5 Flash
# #     # if st.button("Analyze Images"):
# #     with st.spinner("Analyzing images..."):
# #         analysis_results = analyze_images(images)
# #         for idx, result in enumerate(analysis_results):
            
# #             st.write(result)


# if uploaded_file is not None:
#     st.write("Processing your PDF file...")

#     # Process PDF and convert to images
#     images = process_pdf(uploaded_file)
#     st.write(f"{len(images)} images have been extracted from the PDF.")

#     # Analyze images with Gemini 1.5 Flash
#     # if st.button("Analyze Images with Gemini"):
#     with st.spinner("Analyzing images..."):
#         gemini_analysis = analyze_images_gemini(images)
#         for idx, result in enumerate(gemini_analysis):
#             st.json(result)

#     # if st.button("Analyze with Nvidia"):
#     #     # Further analysis with Nvidia Neva-22b
#     #     st.write("Refining analysis with Nvidia Neva-22b model...")
#     #     nvidia_analysis = analyze_images_nvidia(images)
#     #     for idx, result in enumerate(nvidia_analysis):
#     #         st.write(result)

    
#     # Clean up temporary files
#     for image in images:
#         os.remove(image)
        

# # import streamlit as st
# # import os
# # import google.generativeai as genai
# # from langchain_nvidia_ai_endpoints import ChatNVIDIA
# # from pdf2image import convert_from_path
# # import tempfile
# # import shutil
# # import requests
# # import base64

# # from langchain.prompts import PromptTemplate

# # # Load environment variables
# # from dotenv import load_dotenv
# # load_dotenv()

# # # Configure API Keys
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Load models
# # gemini_model_for_image = genai.GenerativeModel("gemini-1.5-flash")




# # # Set up Streamlit page
# # st.set_page_config("PDF to Image Analysis")

# # uploaded_file = st.file_uploader("Upload your PDF file here")




# # def process_pdf(pdf_file):
# #     """Converts PDF to images."""
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
# #         shutil.copyfileobj(pdf_file, tmpfile)
# #         pdf_path = tmpfile.name

# #     pages = convert_from_path(pdf_path, 300)  # 300 DPI for high resolution
# #     images = []
# #     total_pages = len(pages)
# #     progress_bar = st.progress(0)
    
# #     for i, page in enumerate(pages):
# #         image_path = f'output_image_page_{i + 1}.png'
# #         page.save(image_path, 'PNG')
# #         images.append(image_path)
# #         progress_bar.progress((i + 1) / total_pages)
        
# #     return images


# # def encode_image(image_path):
# #      with open(image_path, "rb") as image_file:
# #          return base64.b64encode(image_file.read()).decode('utf-8')

# # def analyze_images_gemini(images):
# #     """Analyze images using the Gemini 1.5 Flash model."""
# #     results = []
# #     for image in images:
# #         with open(image, "rb") as img_file:
# #             # Upload image to Gemini
# #             myfile = genai.upload_file(image)
# #             # Construct prompt for Gemini model
# #             prompt = (
# #                 "You are a technical expert analyzing a blueprint. "
# #                 "Please provide the following details: "
# #                 "1. Dimensions (length, breadth, height) "
# #                 "2. Part number "
# #                 "3. Material "
# #                 "4. Step-by-step manufacturing process"
# #             )
# #             # Generate content from the image
# #             result = gemini_model_for_image.generate_content([myfile, "\n\n", prompt])
# #             results.append(result.text)
# #     return results

# # # def analyze_images_nvidia(images):
# # #     """Use Nvidia Neva-22b for analyzing the images directly."""
# # #     results = []
# # #     api_key = os.getenv("Nvidia")

# # #     for image in images:
# # #         with open(image, "rb") as img_file:
# # #             files = {
# # #                 'file': img_file
# # #             }
# # #             headers = {
# # #                 "Authorization": f"Bearer {api_key}",
# # #                 "Accept": "application/json"
# # #             }

# # #             payload = {
# # #                 "messages": [
# # #                     {
# # #                         "role": "user",
# # #                         "content": "Please analyze this technical drawing. Provide the part number, material, dimensions, and suggested manufacturing processes."
# # #                     }
# # #                 ]
# # #             }

# # #             # Make the API request to Nvidia
# # #             response = requests.post('https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b', headers=headers, json=payload, files=files)  # Add your actual Nvidia image endpoint here.
            
# # #             if response.status_code == 200:
# # #                 results.append(response.json().get('choices')[0].get('message').get('content'))
# # #             else:
# # #                 results.append(f"Error: {response.status_code} - {response.text}")

# # #     return results

# # def analyze_text(result):
# #     text =""
# #     api_key =  "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"
# #     prompt = (
                
# #                 "Please provide the following details using the info provided "
# #                 "1. Dimensions (length, breadth, height) "
# #                 "2. Part number "
# #                 "3. Material "
# #                 "4. Step-by-step manufacturing process"
            
# #     )

# #     client = ChatNVIDIA(
# #     model="meta/llama-3.1-405b-instruct",
# #     api_key =api_key,
# #     temperature=0.2,
# #     top_p=0.7,
# #     max_tokens=1024,
# #     )

# #     for chunk in client.stream([{"role":"user","content":prompt}]): 
# #         text+=chunk.content

# #     return text




    



# # if uploaded_file is not None:
# #     st.write("Processing your PDF file...")

# #     # Process PDF and convert to images
# #     images = process_pdf(uploaded_file)
# #     st.write(f"{len(images)} images have been extracted from the PDF.")

# #     # Analyze images with Gemini 1.5 Flash

# #     with st.spinner("Analyzing images..."):
# #         gemini_analysis = analyze_images_gemini(images)
        


# #         text =analyze_text(gemini_analysis)
# #         st.write(text)
        
                

                
            

# #     # Analyze images with Nvidia Neva-22b
# #     # if st.button("Analyze with Nvidia"):
# #     #     with st.spinner("Refining analysis with Nvidia Neva-22b model..."):
# #     #         nvidia_analysis = analyze_images_nvidia(images)
# #     #         for idx, result in enumerate(nvidia_analysis):
# #     #             st.write(result)
    
# #     # Clean up temporary files
# #     for image in images:
# #         os.remove(image)
# # import os
# # import google.generativeai as genai
# # from pdf2image import convert_from_path
# # import tempfile
# # import shutil
# # import streamlit as st
# # from google.ai.generativelanguage_v1beta.types import content
# # from dotenv import load_dotenv

# # # Load environment variables
# # load_dotenv()

# # # Configure API key
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # Create the generation configuration for the model
# # generation_config = {
# #     "temperature": 1,
# #     "top_p": 0.95,
# #     "top_k": 64,
# #     "max_output_tokens": 8192,
# #     "response_schema": content.Schema(
# #         type=content.Type.OBJECT,
# #         properties={
# #             "manufacturing_process": content.Schema(type=content.Type.STRING),
# #             "length": content.Schema(type=content.Type.NUMBER),
# #             "material": content.Schema(type=content.Type.STRING),
# #             "breadth": content.Schema(type=content.Type.NUMBER),
# #             "height": content.Schema(type=content.Type.NUMBER),
# #         },
# #     ),
# #     "response_mime_type": "application/json",
# # }

# # # Load the Gemini model for image analysis
# # model = genai.GenerativeModel(
# #     model_name="gemini-1.5-flash",
# #     generation_config=generation_config,
# #     system_instruction=(
# #         "You are a technical expert analyzing a blueprint. "
# #         "Extract the following details: dimensions (length, breadth, height), "
# #         "part number, material, and the step-by-step manufacturing process."
# #     ),
# # )

# # # Set up Streamlit page
# # st.set_page_config("PDF to Image Analysis")
# # uploaded_file = st.file_uploader("Upload your PDF file here")

# # def process_pdf(pdf_file):
# #     """Converts PDF to images."""
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
# #         shutil.copyfileobj(pdf_file, tmpfile)
# #         pdf_path = tmpfile.name

# #     pages = convert_from_path(pdf_path, 300)  # 300 DPI for high resolution
# #     images = []
# #     total_pages = len(pages)
# #     progress_bar = st.progress(0)
    
# #     for i, page in enumerate(pages):
# #         image_path = f'output_image_page_{i + 1}.png'
# #         page.save(image_path, 'PNG')
# #         images.append(image_path)
# #         progress_bar.progress((i + 1) / total_pages)
        
# #     return images

# # def upload_to_gemini(path, mime_type=None):
# #     """Uploads the given file to Gemini."""
# #     file = genai.upload_file(path, mime_type=mime_type)

# #     return file

# # def analyze_images(images):
# #     """Analyze multiple images using the Gemini model."""
# #     results = []

# #     # for image_file in images:
# #     # Upload the image to Gemini
# #     uploaded_file1 = upload_to_gemini(images[0], mime_type="image/jpeg")
# #     uploaded_file2 = upload_to_gemini(images[1], mime_type="image/jpeg")

# #     # Construct the prompt for the Gemini model
# #     prompt = (
# #     "You are a highly skilled technical expert specializing in blueprint analysis. "
# #     "Your task is to carefully examine the provided blueprint image and extract detailed information. "
# #     "For each blueprint, please provide the following data points: \n"
# #     "1. **Dimensions**: Specify the length, breadth, and height of the part in millimeters. \n"
# #     "2. **Part Number**: Identify and list the part number associated with the blueprint. \n"
# #     "3. **Material**: Describe the material used for the part, including any specifications if available. \n"
# #     "4. **Manufacturing Process**: Outline the step-by-step manufacturing process involved in creating this part, detailing any critical steps. \n"
# #     "For example: \n"
# #     "Length: 100mm, Breadth: 50mm, Height: 20mm, Part Number: ABC123, Material: Aluminum, Manufacturing Process: Cutting, Drilling, Finishing. \n"
# #     "Ensure that your response is clear, structured, and includes all requested details. If any information is not visible, please indicate that."
# #     )



# #     # Call the model to generate content based on the image
# #     response = model.generate_content([uploaded_file1,uploaded_file2, "\n\n", prompt])
# #     results.append(response.text)

# #     return results

# # if uploaded_file is not None:
# #     st.write("Processing your PDF file...")

# #     # Process PDF and convert to images
# #     images = process_pdf(uploaded_file)
# #     st.write(f"{len(images)} images have been extracted from the PDF.")

# #     # Analyze images with Gemini 1.5 Flash
# #     with st.spinner("Analyzing images..."):
# #         gemini_analysis = analyze_images(images)
       
# #         for idx, result in enumerate(gemini_analysis):
# #             st.write(f"Analysis result for image {idx + 1}:")
# #             st.json(result)  # Display structured result as JSON

# #     # Clean up temporary images
# #     for image in images:
# #         os.remove(image)
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
    "max_output_tokens": 8192,
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
        "Make sure to specify which view (top, front, side) you are extracting these from, and ensure consistency."
    )
    result = gemini_model_for_dimensions.generate_content([uploaded_file, prompt])
    results.append(result.text)
    return results

def analyze_text(images):
    results = []
    
    uploaded_file1 = upload_to_gemini(images[0])
    uploaded_file2 = upload_to_gemini(images[1])
    prompt = (
        "Please extract the part number, material used, and step-by-step manufacturing process from the blueprint. "
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
