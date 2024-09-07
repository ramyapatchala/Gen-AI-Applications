import streamlit as st
from openai import OpenAI, OpenAIError
import fitz  # PyMuPDF for reading PDFs
import requests
import cohere  # For Cohere

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

# Sidebar: Provide the user with LLM selection options.
st.sidebar.header("LLM Options")
llm_option = st.sidebar.selectbox(
    "Choose an LLM to use for generating the summary:",
    options=["OpenAI (GPT-4)", "Cohere"]
)

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

# Sidebar: Provide a dropdown menu for language selection.
language_option = st.sidebar.selectbox(
    "Choose the output language:",
    options=["English", "French", "Spanish"]
)

# Let the user enter a URL for the PDF.
url = st.text_input("Enter the URL to the PDF:")

# Initialize document variable.
document = None

# Handle URL input if provided.
if url:
    document = read_pdf_from_url(url)

# LLM Key validation section.
openai_api_key = st.secrets.get('key1')  # OpenAI API key
cohere_api_key = st.secrets.get('cohere_key')  # Cohere API key

# Validate keys based on selected LLM.
valid_key = False
if llm_option == "OpenAI (GPT-4)" and openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        client.models.list()  # Simple API call to verify key.
        st.sidebar.success("OpenAI key is valid!", icon="✅")
        valid_key = True
    except OpenAIError as e:
        st.sidebar.error(f"Invalid OpenAI key: {e}", icon="❌")
elif llm_option == "Cohere" and cohere_api_key:
    try:
        cohere_client = cohere.Client(api_key=cohere_api_key)
        cohere_client.models.list()   # Simple API call to verify key.
        st.sidebar.success("Cohere key is valid!", icon="✅")
        valid_key = True
    except Exception as e:
        st.sidebar.error(f"Invalid Cohere key: {e}", icon="❌")

# Proceed if the key is valid.
if valid_key and document:
    # Modify the summary instruction based on the user's selection.
    if summary_option == "Summarize the document in 100 words":
        summary_instruction = "Summarize in 100 words."
    elif summary_option == "Summarize in 2 concise paragraphs":
        summary_instruction = "Summarize this document in 2 connecting paragraphs."
    else:
        summary_instruction = "Summarize in 5 bullet points."
    
    # Adjust the prompt to include the chosen language.
    if language_option == "English":
        language_instruction = "Please summarize."
    elif language_option == "French":
        language_instruction = "Veuillez résumer."
    else:
        language_instruction = "Por favor, resuma."

    # Combine the document and summary instructions, including the language.
    prompt = f"Here's a document: {document} \n\n---\n\n {summary_instruction} {language_instruction}"

    # Generate an answer using the selected LLM.
    try:
        if llm_option == "OpenAI (GPT-4)":
            response = client.chat.completions.create(
                model="gpt-4",  # You can use GPT-4 or any other model you prefer.
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            st.write_stream(response)  # Stream OpenAI's response.
        elif llm_option == "Cohere":
            cohere_response = cohere_client.generate(
                prompt=prompt
            )
            st.write(cohere_response.generations[0].text)  # Display the Cohere response.
    except Exception as e:
        st.error(f"Error generating summary: {e}", icon="❌")

# Reset document if no URL is provided.
if not url:
    st.info("Please enter a valid PDF URL to continue.", icon="🌐")
