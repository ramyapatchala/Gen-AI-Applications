import streamlit as st
from openai import OpenAI
import os

page1 = st.Page("HWs/HW1.py", title="HW 1: Document Q&A System")
page2 = st.Page("HWs/HW2.py", title="HW 2: Multi Webpage Summarizer")
page3 = st.Page("HWs/HW3.py", title="HW 3: Chatbot")
page4 = st.Page("HWs/HW4.py", title="HW 4: Retreived Augmented Generation (RAG)")
page5 = st.Page("HWs/HW5.py", title="HW 5: iSchool Club Chatbot")
page6 = st.Page("HWs/HW6.py", title = "HW 6: News Bot")

pg = st.navigation([page1, page2, page3, page4, page5, page6])
st.set_page_config(page_title="HW Manager")
pg.run()
