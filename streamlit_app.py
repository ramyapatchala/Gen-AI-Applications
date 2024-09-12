import streamlit as st
from openai import OpenAI
import os

page1 = st.Page("HWs/HW1.py", title="HW 1")
page2 = st.Page("HWs/HW2.py", title="HW 2")
page3 = st.Page("HWs/HW3.py", title="HW 3")

pg = st.navigation([page1, page2, page3])
st.set_page_config(page_title="HW Manager")
pg.run()
