import streamlit as st
from openai import OpenAI, OpenAIError
import fitz  # PyMuPDF for reading PDFs

def read_pdf(file):
    """Function to read PDF content using PyMuPDF (fitz)."""
    document = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            document += page.get_text()
    return document

# Set the page configuration at the very beginning of the script.
st.set_page_config(page_title="Document Q&A")

# Show title and description.
st.title("📄 Document Question Answering - Q&A")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("OpenAI API Key", type="password")

# Validate API key as soon as it's entered.
if openai_api_key:
    try:
        # Create an OpenAI client to validate the key.
        client = OpenAI(api_key=openai_api_key)
        # Try a simple API call to check if the key is valid
        client.models.list()
        st.success("API key is valid!", icon="✅")
    except OpenAIError as e:
        st.error(f"Invalid API key: {e}", icon="❌")
else:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")

# Proceed if API key is provided and valid
if openai_api_key and 'client' in locals():
    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # Initialize document variable
    document = None

    # Check if a file is uploaded
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Read the uploaded file based on its type
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")

        # Check if the document is loaded successfully
        if document:
            # Ask the user for a question via `st.text_area`.
            question = st.text_area(
                "Now ask a question about the document!",
                placeholder="Can you give me a short summary?",
                disabled=False,
            )

            if question:
                # Process the uploaded file and question.
                messages = [
                    {
                        "role": "user",
                        "content": f"Here's a document: {document} \n\n---\n\n {question}",
                    }
                ]

                try:
                    # Generate an answer using the OpenAI API.
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        stream=True,
                    )

                    # Stream the response to the app using `st.write`.
                    st.write_stream(response)
                except OpenAIError as e:
                    st.error(f"An error occurred while generating a response: {e}", icon="❌")

    # Reset document if the file is removed
    if not uploaded_file:
        document = None
        st.info("Please upload a document to continue.", icon="📄")
