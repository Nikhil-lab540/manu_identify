import streamlit as st
import os

from streamlit_pdf_viewer import pdf_viewer
# Set up Streamlit page

st.set_page_config("PDF to Image Analysis")
from streamlit import session_state as ss


# Access the uploaded ref via a key.
st.file_uploader("Upload PDF file", type=('pdf'), key='pdf')

if ss.pdf:
    ss.pdf_ref = ss.pdf  # backup

# Now you can access "pdf_ref" anywhere in your app.
if ss.pdf_ref:
    binary_data = ss.pdf_ref.getvalue()
    pdf_viewer(input=binary_data, width=700)