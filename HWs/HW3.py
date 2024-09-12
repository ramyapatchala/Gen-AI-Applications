import streamlit as st
import google.generativeai as genai

st.title("My lab3 Question answering chatbot")

# Initialize the Gemini client
if 'client' not in st.session_state:
    api_key = st.secrets['gemini_key']
    genai.configure(api_key=api_key)
    st.session_state.client = genai.GenerativeModel('gemini-pro')

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "model", "parts": [{"text": "How can I help you?"}]}]

# Display chat history
for msg in st.session_state.messages:
    chat_msg = st.chat_message("assistant" if msg["role"] == "model" else "user")
    chat_msg.write(msg["parts"][0]["text"])

# Chat input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "parts": [{"text": prompt}]})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare chat history for Gemini API
    chat_history = st.session_state.messages[:-1]  # Exclude the last message
    
    # Generate response using Gemini API
    client = st.session_state.client
    try:
        response = client.generate_content(
            contents=[*chat_history, {"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=1500,
            ),
            stream=True
        )
        
        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "model", "parts": [{"text": full_response}]})
        
        # Limit chat history to last 5 messages
        st.session_state.messages = st.session_state.messages[-5:]
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
