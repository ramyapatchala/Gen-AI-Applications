import os
import streamlit as st
from mistralai import Mistral

import time
import requests
import fitz  # PyMuPDF for reading PDFs


mistral_key = st.secrets['mistral_key']


# Function to read PDF content from a URL using PyMuPDF (fitz).
def read_pdf_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            document = ""
            for page in doc:
                document += page.get_text()
        return document
    except requests.RequestException as e:
        st.error(f"Error fetching PDF from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# Streamlit app title and description.
st.title("📄 PDF Summarizer from URL")
st.write("Enter a PDF URL and select summary options below.")

# Function to validate Cohere API key.
def verify_mistral_key(api_key):
    client = Mistral(api_key=api_key)
    try:
        # Perform a test call
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": "Test message"}
            ]
        )
        # Check if response contains expected data
        if response.choices:
            return True, "API key is valid"
        else:
            return False, "API key is invalid or no response received"
    except Exception as e:
        return False, str(e)

# Verify the Mistral API key
is_valid, message = verify_mistral_key(mistral_key)

if is_valid:
    st.success(message)
else:
    st.error(f"Invalid Mistral API key: {message}")


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

# Sidebar: Provide a dropdown menu for language selection
language_option = st.sidebar.selectbox(
    "Choose the output language:",
    options=["English", "French", "Spanish"]
)

# Let the user enter a URL for the PDF
url = st.text_input("Enter the URL to the PDF:")

# Ensure the `url` variable is properly initialized
if url:
    # Initialize document variable
    document = read_pdf_from_url(url)

    # If document is successfully loaded from URL
    if document:
        # Modify the prompt based on the selected summary option.
        if summary_option == "Summarize the document in 100 words":
            summary_instruction = "Summarize this document in 100 words."
        elif summary_option == "Summarize the document in 2 connecting paragraphs":
            summary_instruction = "Summarize this document in 2 connecting paragraphs."
        else:
            summary_instruction = "Summarize this document in 5 bullet points."

        # Adjust the prompt to include the chosen language
        if language_option == "English":
            language_instruction = "Please summarize the document in English."
        elif language_option == "Spanish":
            language_instruction = "Por favor, resuma el documento en español."
        else:
            language_instruction = "Veuillez résumer le document en français."

        # Combine document, summary, and language instructions.
        prompt = f"{language_instruction} {summary_instruction} \n\n\n---\n\n Document: {document}\n\n---\n\n"
        try:    
            model = "mistral-large-latest"
            client = Mistral(api_key=mistral_key)
            chat_response = client.chat.complete(
                    model= model,
                    messages = [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                )
            st.write(chat_response.choices[0].message.content)

        except Exception as e:
            st.error(f"Unexpected error: {e}", icon="❌")
else:
    st.info("Please enter a valid PDF URL to generate a summary.", icon="🌐")
