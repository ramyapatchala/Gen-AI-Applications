import time
import requests
import fitz  # PyMuPDF for reading PDFs
import cohere
import streamlit as st

# Load Cohere API key from environment variables
cohere_key = st.secrets.get('cohere_key')

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

# Validate Cohere API key.
if cohere_key:
    try:
        co = cohere.Client(cohere_key)
        # Tokenize the system message (not necessary for this use case).
        # response = co.tokenize(text=system_message, model='command')
        st.sidebar.success("Cohere API key is valid!", icon="✅")
    except Exception as e:
        st.error(f"Invalid Cohere API key: {e}", icon="❌")
else:
    st.error("Cohere API key not found in secrets!", icon="❌")

# Summary options.
summary_options = {
    "100_words": "Summarize in 100 words",
    "2_paragraphs": "Summarize in 2 connecting paragraphs",
    "5_bullet_points": "Summarize in 5 bullet points"
}

# Sidebar: Provide the user with summary options.
st.sidebar.header("Summary Options")

summary_option = st.sidebar.selectbox(
    "Select a summary style:",
    options=list(summary_options.keys()),  # Use keys as options
    format_func=lambda x: summary_options[x]
)


# Language selection.
language_options = {
    "english": "English",
    "french": "French",
    "spanish": "Spanish"
}

# Sidebar: Provide a dropdown menu for language selection
language_option = st.sidebar.selectbox(
    "Choose output language:",
    options=list(language_options.keys()),  # Use keys as options
    format_func=lambda x: language_options[x]
)


# User input for PDF URL.
url = st.text_input("Enter PDF URL:")

# Initialize document variable.
document = None

# Fetch and process the PDF content when URL is provided.
if url:
    document = read_pdf_from_url(url)

# Generate summary if document is available.
if document:
    # Construct the summary instruction based on user selection.
    if summary_option == summary_options['100_words']:
        summary_instruction = "Summarize in 100 words."
    elif summary_option == summary_options['2_paragraphs']:
        summary_instruction = "Summarize in 2 connecting paragraphs."
    else:
        summary_instruction = "Summarize in 5 bullet points."

    # Construct the language instruction.
    language_instruction = {
        "English": "Please summarize in English.",
        "French": "Veuillez résumer en français.",
        "Spanish": "Por favor, resuma en español."
    }[language_option]

    # Combine document, summary, and language instructions.
    prompt = f"Document: {document}\n\n---\n\n{summary_instruction} {language_instruction}"

    try:
        # Generate summary using Cohere
        response = co.chat_stream(
            model='command-r',
            message=prompt,
            temperature=0,
            max_tokens=1500,
            chat_history=[{"role": "SYSTEM", "message": system_message}],
            prompt_truncation='AUTO',
            connectors=[],
            documents=[]
        )

        response_text = ""
        for event in response:
            if event.event_type == "text-generation":
                response_text += event.text
        st.write(response_text)

    except Exception as e:
        st.error(f"Error generating summary: {e}", icon="❌")

# Display message if no URL is provided.
if not url:
    st.info("Please enter a valid PDF URL to generate a summary.", icon="🌐")
