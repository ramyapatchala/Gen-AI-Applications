import streamlit as st
from openai import OpenAI, OpenAIError
import fitz  # PyMuPDF for reading PDFs
import requests
from bs4 import BeautifulSoup

# Function to read PDF files from a URL
def read_pdf_from_url(url):
    """Function to fetch and read PDF content from a URL using PyMuPDF (fitz)."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            document = ""
            for page in doc:
                document += page.get_text()
        return document
    except requests.RequestException as e:
        st.error(f"Error reading PDF from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing the PDF: {e}")
        return None

# Show title and description.
st.title("📄 PDF Summarizer from URL")
st.write(
    "Enter a PDF URL below and ask for a summary."
)

# Use the OpenAI API key stored in Streamlit secrets
openai_api_key = st.secrets['key1']

# Validate the API key from secrets.
if openai_api_key:
    try:
        # Create an OpenAI client using the API key from secrets
        client = OpenAI(api_key=openai_api_key)
        # Try a simple API call to check if the key is valid
        client.models.list()
        st.success("API key is valid!", icon="✅")
    except OpenAIError as e:
        st.error(f"Invalid API key: {e}", icon="❌")
else:
    st.error("API key not found in secrets!", icon="❌")

# Proceed if API key is provided and valid
if openai_api_key and 'client' in locals():
    
    # Sidebar: Provide the user with summary options.
    st.sidebar.header("Summary Options")
    
    summary_option = st.sidebar.radio(
        "Choose how you would like the document to be summarized:",
        options=[
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points"
        ]
    )
    
    # Sidebar: Checkbox for selecting the model
    use_advanced_model = st.sidebar.checkbox("Use Advanced Model")
    
    # Choose model based on the checkbox
    model_choice = "gpt-4o" if use_advanced_model else "gpt-4o-mini"
    
    # Let the user enter a URL for the PDF
    url = st.text_input("Enter the URL to the PDF:")

    # Initialize document variable
    document = None

    # Handle URL input if provided
    if url:
        document = read_pdf_from_url(url)

    # If document is successfully loaded from URL
    if document:
        # Modify the question based on the selected summary option.
        if summary_option == "Summarize the document in 100 words":
            summary_instruction = "Summarize this document in 100 words."
        elif summary_option == "Summarize the document in 2 connecting paragraphs":
            summary_instruction = "Summarize this document in 2 connecting paragraphs."
        else:
            summary_instruction = "Summarize this document in 5 bullet points."
        
        # Combine the document and summary instruction
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {summary_instruction}",
            }
        ]
        
        # Generate an answer using the OpenAI API with the selected model.
        try:
            stream = client.chat.completions.create(
                model=model_choice,
                messages=messages,
                stream=True,
            )
            
            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)
        
        except OpenAIError as e:
            st.error(f"Error generating summary: {e}", icon="❌")

    # Reset document if no URL is provided
    if not url:
        st.info("Please enter a valid PDF URL to continue.", icon="🌐")
