import streamlit as st
from openai import OpenAI, OpenAIError
import fitz  # PyMuPDF for reading PDFs
import requests
import cohere

# Ensure CohereError is imported correctly.
from cohere import CohereError

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
st.write("Enter a PDF URL and customize your summary options below.")

# Sidebar: LLM and Summary Options
st.sidebar.header("LLM and Summary Settings")

# LLM selection.
llm_option = st.sidebar.selectbox(
    "Choose an LLM for summarization:",
    options=["OpenAI (GPT-4)", "Cohere"]
)

# Summary options.
summary_options = {
    "100_words": "Summarize in 100 words",
    "2_paragraphs": "Summarize in 2 connecting paragraphs",
    "5_bullet_points": "Summarize in 5 bullet points"
}

summary_option = st.sidebar.selectbox(
    "Select a summary style:",
    options=summary_options.values(),
    format_func=lambda x: summary_options[x]
)

# Language selection.
language_options = {
    "english": "English",
    "french": "French",
    "spanish": "Spanish"
}

language_option = st.sidebar.selectbox(
    "Choose output language:",
    options=language_options.values(),
    format_func=lambda x: language_options[x]
)

# User input for PDF URL.
url = st.text_input("Enter PDF URL:")

# Initialize document variable.
document = None

# Fetch and process the PDF content when URL is provided.
if url:
    document = read_pdf_from_url(url)

# LLM Key validation and initialization.
openai_api_key = st.secrets.get('key1')
cohere_api_key = st.secrets.get('cohere_key')

# Validate and initialize the selected LLM.
if llm_option == "OpenAI (GPT-4)" and openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        openai_client.models.list()
        st.sidebar.success("OpenAI key is valid!", icon="✅")
    except OpenAIError as e:
        st.sidebar.error(f"Invalid OpenAI key: {e}", icon="❌")
elif llm_option == "Cohere" and cohere_api_key:
    try:
        cohere_client = cohere.Client(api_key=cohere_api_key)
        cohere_client.check_token()
        st.sidebar.success("Cohere key is valid!", icon="✅")
    except CohereError as e:
        st.sidebar.error(f"Invalid Cohere key: {e}", icon="❌")

# Generate summary if LLM is initialized and document is available.
if document and ('openai_client' in locals() or 'cohere_client' in locals()):
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
        if llm_option == "OpenAI (GPT-4)":
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            st.write_stream(response)
        elif llm_option == "Cohere":
            cohere_response = cohere_client.generate(
                prompt=prompt,
                max_tokens=300
            )
            st.write(cohere_response.generations[0].text)
    except Exception as e:
        st.error(f"Error generating summary: {e}", icon="❌")

# Display message if no URL is provided.
if not url:
    st.info("Please enter a valid PDF URL to generate a summary.", icon="🌐")
