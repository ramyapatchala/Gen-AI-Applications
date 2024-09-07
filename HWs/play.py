import time
import os
import logging
from dotenv import load_dotenv
import fitz  # PyMuPDF for reading PDFs
import requests
import cohere
import streamlit as st
cohere_key = st.secrets['cohere_key']

load_dotenv("test.env")
logging.basicConfig(filename = 'test_models_debug.txt', level=logging.INFO)
question_to_ask = "Why are LLMs (AI) a danger to society?"

system_message = """
Goal: Answer the question using bullets.
      The answer should be appropriate for a 10 year old child to understand
      """
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
st.write("Enter a PDF URL below and select your preferred language for the summary.")

# Validate the API key from secrets.
if cohere_key:
    try:
        # Create an OpenAI client using the API key from secrets
        client = Cohere(api_key=cohere_key)
        # Try a simple API call to check if the key is valid
        client.models.list()
        st.success("Cohere API key is valid!", icon="✅")
    except OpenAIError as e:
        st.error(f"Invalid Cohere API key: {e}", icon="❌")
else:
    st.error("API key not found in secrets!", icon="❌")

# Proceed if API key is provided and valid
if cohere_key and 'client' in locals():
    
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
    #use_advanced_model = st.sidebar.checkbox("Use Advanced Model")
    
    # Choose model based on the checkbox
    #model_choice = "gpt-4o" if use_advanced_model else "gpt-4o-mini"
    
    # Sidebar: Provide a dropdown menu for language selection
    language_option = st.sidebar.selectbox(
        "Choose the output language:",
        options=["English", "French", "Spanish"]
    )
    
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
        
        # Adjust the prompt to include the chosen language
        if language_option == "English":
            language_instruction = "Please summarize the document in English."
        elif language_option == "French":
            language_instruction = "Veuillez résumer le document en français."
        else:
            language_instruction = "Por favor, resuma el documento en español."

    question_to_ask = f"Here's a document: {document} \n\n---\n\n {summary_instruction} {language_instruction}"
    co = cohere.Client(cohere_key)
    response = co.tokenize(
        text = system_message, model='command')
    events = co.chat_stream(
                    model='command-r',
                    message=question_to_ask,
                    temperature=0,       
                    max_tokens=1500,
                    chat_history=[{"role":"SYSTEM", "message":system_message}],
                    prompt_truncation='AUTO',
                    connectors=[],
                    documents=[]
    )

    response_text=""
    for event in events:
        if event.event_type=="text-generation":
            response_text = response_text + str(event.text)
    st.write(response_text)
              
    # Reset document if no URL is provided
    if not url:
        st.info("Please enter a valid PDF URL to continue.", icon="🌐")
