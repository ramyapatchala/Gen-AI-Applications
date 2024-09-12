import streamlit as st
import cohere

st.title("My lab3 Question answering chatbot")

# Initialize the Cohere client
if 'client' not in st.session_state:
    api_key = st.secrets['cohere_key']
    st.session_state.client = cohere.Client(api_key)

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "CHATBOT", "content": "How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    chat_msg = st.chat_message("assistant" if msg["role"] == "CHATBOT" else "user")
    chat_msg.write(msg["content"])

# Chat input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "USER", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare chat history for Cohere API
    chat_history = [
        {"role": msg["role"], "message": msg["content"]}
        for msg in st.session_state.messages[:-1]  # Exclude the last message
    ]
    
    # Generate response using Cohere API
    client = st.session_state.client
    try:
        stream = client.chat_stream(  # Changed from chat() to chat_stream()
            model='command',
            message=prompt,
            chat_history=chat_history,
            temperature=0,       
            max_tokens=1500
            # Removed stream=True parameter
        )
        
        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for event in stream:
                if event.event_type == "text-generation":
                    full_response += event.text
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "CHATBOT", "content": full_response})
        
        # Limit chat history to last 5 messages
        st.session_state.messages = st.session_state.messages[-5:]
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
