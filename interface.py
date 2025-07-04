import streamlit as st  # This must be the very first import
st.set_page_config(page_title="AI Text Summarizer", layout="centered")  # Call this immediately after importing streamlit

from transformers import pipeline
import PyPDF2
from PIL import Image
import pytesseract
import io

# Load summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

summarizer = load_summarizer()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to extract text from image
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# Function to summarize text
def get_summary(text, min_len=30, max_len=100):
    if len(text.split()) < 30:
        return "Text too short to summarize."
    
    # Split large text into chunks if it's too long
    chunk_size = 1000  # Adjust this chunk size based on your use case
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    summary = ""
    for chunk in text_chunks:
        summary_chunk = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
        summary += summary_chunk[0]['summary_text'] + "\n"
    
    return summary

# Streamlit App UI
st.title("📝 Multi-Input AI Text Summarizer")
st.markdown("Summarize content from **Text**, **PDF**, or **Image** in seconds!")

tab1, tab2, tab3 = st.tabs(["✏️ Text", "📄 PDF", "🖼️ Image"])

with tab1:
    input_text = st.text_area("Enter or paste your text here:")
    if st.button("Summarize Text"):
        if input_text.strip():
            summary = get_summary(input_text)
            st.success(summary)
        else:
            st.warning("Please enter some text.")

with tab2:
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
    if st.button("Summarize PDF") and uploaded_pdf is not None:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_pdf)
            if extracted_text:
                summary = get_summary(extracted_text)
                st.success(summary)
            else:
                st.warning("No text found in this PDF or it may be a scanned image.")

with tab3:
    uploaded_img = st.file_uploader("Upload an image with text", type=["png", "jpg", "jpeg"])
    if st.button("Summarize Image Text") and uploaded_img is not None:
        with st.spinner("Extracting text from image..."):
            extracted_text = extract_text_from_image(uploaded_img)
            if extracted_text:
                summary = get_summary(extracted_text)
                st.success(summary)
            else:
                st.warning("No text found in this image.")
