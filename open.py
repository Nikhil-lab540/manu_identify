import streamlit as st
import os
import google.generativeai as genai
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pdf2image import convert_from_path
import tempfile
import shutil
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Tool,
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure API Keys
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["NVIDIA_API_KEY"] = os.getenv("Nvidia")

# Define function declarations
def extract_blueprint_details(dimensions, part_number, material, manufacturing_process):
    """Function to extract details from blueprints."""
    return {
        "dimensions": dimensions,
        "part_number": part_number,
        "material": material,
        "manufacturing_process": manufacturing_process,
    }

# Set up the tool for the generative model
tool = Tool(
    function_declarations=[FunctionDeclaration(
        name="extract_blueprint_details",
        description="Extract dimensions, part number, material, and manufacturing process from a blueprint.",
        parameters={
            "type": "object",
            "properties": {
                "dimensions": {
                    "type": "object",
                    "properties": {
                        "length": {"type": "number", "description": "Length of the part in millimeters"},
                        "breadth": {"type": "number", "description": "Breadth of the part in millimeters"},
                        "height": {"type": "number", "description": "Height of the part in millimeters"},
                    },
                    "required": ["length", "breadth", "height"],
                },
                "part_number": {"type": "string", "description": "Part number of the blueprint"},
                "material": {"type": "string", "description": "Material used for the part"},
                "manufacturing_process": {"type": "string", "description": "Step-by-step manufacturing process"},
            },
            "required": ["dimensions", "part_number", "material", "manufacturing_process"],
        }
    )]
)

# Load models with function calling
gemini_model = GenerativeModel(
    "gemini-1.5-flash",
    tools=[tool]
)

# Load LLM
nvidia_llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

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
    total_pages = len(pages)
    progress_bar = st.progress(0)

    for i, page in enumerate(pages):
        image_path = f'output_image_page_{i + 1}.png'
        page.save(image_path, 'PNG')
        images.append(image_path)
        progress_bar.progress((i + 1) / total_pages)

    return images

def analyze_images_gemini(images):
    """Analyze images using the Gemini 1.5 Flash model with function calling for structured output."""
    results = []

    for image in images:
        try:
            with open(image, "rb") as img_file:
                # Upload image to Gemini
                myfile = genai.upload_file(image)

                # Construct prompt for Gemini model
                prompt = (
                    "You are a technical expert analyzing a blueprint. "
                    "Please provide the following details: "
                    "1. Dimensions (length, breadth, height) "
                    "2. Part number "
                    "3. Material "
                    "4. Step-by-step manufacturing process"
                )

                # Generate content from the image
                response = gemini_model.generate_content([myfile, "\n\n", prompt])

                # Check if the response has choices and process function calling
                if response.get("choices"):
                    function_call_output = response["choices"][0]["message"]
                    if function_call_output.get("function_call"):
                        # Extract arguments from function call
                        args = function_call_output["function_call"]["arguments"]
                        if isinstance(args, dict):  # Ensure it's a dictionary
                            structured_data = extract_blueprint_details(**args)
                            results.append(structured_data)
                        else:
                            results.append({"error": "Function call arguments are not in expected format."})
                    else:
                        # If no function call, append raw response
                        results.append(function_call_output)

        except Exception as e:
            st.error(f"An error occurred while analyzing {image}: {str(e)}")
            results.append({"image": image, "error": str(e)})

    return results

if uploaded_file is not None:
    st.write("Processing your PDF file...")

    # Process PDF and convert to images
    images = process_pdf(uploaded_file)
    st.write(f"{len(images)} images have been extracted from the PDF.")

    # Analyze images with Gemini 1.5 Flash
    with st.spinner("Analyzing images..."):
        gemini_analysis = analyze_images_gemini(images)
        st.write("Analysis Results:")
        for result in gemini_analysis:
            if 'error' in result:
                st.error(f"Error processing image: {result['image']} - {result['error']}")
            else:
                st.json(result)  # Display structured data in JSON format for clarity

    # Clean up temporary files
    for image in images:
        os.remove(image)
