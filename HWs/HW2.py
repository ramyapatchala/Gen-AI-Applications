import streamlit as st
from openai import OpenAI, OpenAIError
import fitz  # PyMuPDF for reading PDFs
import requests
from bs4 import BeautifulSoup

# Function to read PDF files
def read_pdf(file):
    """Function to read PDF content using PyMuPDF (fitz)."""
    document = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            document += page.get_text()
    return document

# Function to read content from a URL
def read_url_content(url):
    """Function to fetch and extract text from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Show title and description.
st.title("📄 Document and URL Question Answering - Q&A")
st.write(
    "Upload a document below or enter a URL, then ask a question or generate a summary."
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
    
    # Let the user enter a URL or upload a file
    st.header("Enter a URL or Upload a Document")

    # Option to input URL
    url = st.text_input("Enter the URL to summarize:")
    
    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .md, or .pdf)", type=("txt", "md", "pdf")
    )
    
    # Initialize document variable
    document = None

    # Handle URL input if provided
    if url:
        document = read_url_content(url)

    # Handle file upload if provided
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Read the uploaded file based on its type
        if file_extension == 'txt' or file_extension == 'md':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")

    # If document is successfully loaded from URL or upload
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

    # Reset document if neither file nor URL is provided
    if not uploaded_file and not url:
        st.info("Please upload a document or enter a URL to continue.", icon="📄")
