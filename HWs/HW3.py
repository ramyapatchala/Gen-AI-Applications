import streamlit as st
from openai import OpenAI
import tiktoken  # Required for token counting

st.title("LAB 3: Chatbot")

# Sidebar options for model and buffer selection
openAI_model = st.sidebar.selectbox("Which model?", ("mini", "regular"))
buffer_type = st.sidebar.selectbox("Select buffer type", ("Message-based", "Token-based"))

# Determine which model to use
if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

# Initialize the OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets['key1']
    st.session_state.client = OpenAI(api_key=api_key)

# Initialize the message history and conversation state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "Hi there! What cool thing do you want to learn about today?"}]
    st.session_state['wants_more_info'] = False
    st.session_state['last_question'] = ""

# Tokenizer setup for token counting
encoding = tiktoken.encoding_for_model(model_to_use)

def calculate_tokens(messages):
    """Calculate total tokens for a list of messages."""
    total_tokens = 0
    for msg in messages:
        total_tokens += len(encoding.encode(msg['content']))
    return total_tokens

def truncate_messages_by_tokens(messages, max_tokens):
    """Truncate the message buffer to ensure it stays within max_tokens."""
    total_tokens = calculate_tokens(messages)
    while total_tokens > max_tokens and len(messages) > 1:
        messages.pop(0)
        total_tokens = calculate_tokens(messages)
    return messages

# Maximum tokens allowed to send to the model
max_tokens = 4096

# Display previous messages
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything!"):
    # Append user's input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # If the bot is waiting for a 'yes' or 'no' to provide more info
    if st.session_state['wants_more_info']:
        if prompt.lower() == "yes":
            # Provide more info about the last question
            client = st.session_state.client
            more_info_prompt = f"Provide more fun details about this topic for a 10-year-old: {st.session_state['last_question']}"
            more_info_messages = [{"role": "system", "content": "You are a fun, friendly chatbot for 10-year-old kids. Use simple words, short sentences, and make your explanations exciting and easy to understand. Keep your answers brief and engaging."},
                                  {"role": "user", "content": more_info_prompt}]
            
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=more_info_messages,
                stream=True
            )

            with st.chat_message("assistant"):
                more_info_response = st.write_stream(stream)

            st.session_state.messages.append({"role": "assistant", "content": more_info_response})
            
            # Always ask if they want more info
            follow_up_question = "DO YOU WANT MORE INFO?"
            st.session_state.messages.append({"role": "assistant", "content": follow_up_question})
            with st.chat_message("assistant"):
                st.markdown(follow_up_question)
            st.session_state['wants_more_info'] = True
        elif prompt.lower() == "no":
            new_question_prompt = "What other question can I help you with?"
            st.session_state.messages.append({"role": "assistant", "content": new_question_prompt})
            with st.chat_message("assistant"):
                st.markdown(new_question_prompt)
            st.session_state['wants_more_info'] = False
            st.session_state['last_question'] = ""
        else:
            st.warning("Oops! Please say 'yes' or 'no'.")
    else:
        # Regular question handling
        if buffer_type == "Message-based":
            truncated_messages = st.session_state.messages[-2:]
        elif buffer_type == "Token-based":
            truncated_messages = truncate_messages_by_tokens(st.session_state.messages, max_tokens)

        # Add instruction for kid-friendly responses
        kid_friendly_instruction = {"role": "system", "content": "You are a fun, friendly chatbot for 10-year-old kids. Use simple words, short sentences, and make your explanations exciting and easy to understand. Keep your answers brief and engaging."}
        truncated_messages = [kid_friendly_instruction] + truncated_messages

        client = st.session_state.client
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=truncated_messages,
            stream=True
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Store the last question for potential follow-up
        st.session_state['last_question'] = prompt

        # Always ask if they want more information
        st.session_state['wants_more_info'] = True
        follow_up_question = "DO YOU WANT MORE INFO?"
        st.session_state.messages.append({"role": "assistant", "content": follow_up_question})
        with st.chat_message("assistant"):
            st.markdown(follow_up_question)

    # Apply the chosen buffer strategy
    if buffer_type == "Message-based":
        st.session_state.messages = st.session_state.messages[-5:]
    elif buffer_type == "Token-based":
        st.session_state.messages = truncate_messages_by_tokens(st.session_state.messages, max_tokens)