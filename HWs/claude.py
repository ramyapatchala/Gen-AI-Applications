import anthropic
import streamlit as st
question = "Why are LLMs danger to society?"
def do_anthropic(model_to_use):
  api_key = st.secrets['claude_key']
  if model_to_use == 'sonnet':
    model= 'claude-3-sonnet-20240229"
  elif model_to_use == 'haiku':
    model='claude-3-haiku-20230307'
  else:
    model = 'claude-3-opus-20240229'
  client = anthropic.Anthropic(api_key = api_key)

msg_to_llm=[
  {"role":"user", "content":[{"type":"text", "text":question}]}
]
msg = client.messages.create(
  model = model,
  max_tokens = 1500,
  temperature = 0,
  messages = msg_to_llm
)
data = message.context[0].text

return data
