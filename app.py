import base64
import os
import time
import streamlit as st
from PIL import Image
from transformers import pipeline

def zero_shot_classification(text, progress):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["Criminal Activity", "Safe and Ethical"]
    
    for i in range(0, 101, 10):
        progress.progress(i)
        time.sleep(0.1)

    result = classifier(text, labels)
    return result["labels"][0]

st.set_page_config(
    page_title="Verbal Vanguard",
    # page_icon=Image.open("./assets/my_photo.png")
)

def custom_css(css: str):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

css = f"""
body, h1, h2, h3, h4, h5, h6, p, div, a, span, label, input, button, textarea {{
    color: white !important;
}}

::placeholder {{
    color: white !important;
    opacity: 1 !important;
}}

div.stNavBar {{
    background-color: #222 !important;
}}

div.stButton {{
    background-color: #1B0099 !important;
    color: #fff !important;
    border-color: #fff !important;
}}

div.stButton button:hover {{
    background-color: #1573F8 !important;
    color: #fff !important;
    border-color: #fff !important;
}}

div.stButton > button:first-child {{ border: 1px solid #FFF; border-radius:20px 20px 20px 20px; background: none;}}

div.stButton > button:first-child:hover {{
    background: #1B0099;
    color: white;
}}

h1, h3, span {{
    color: #fff;
}}

footer, header {{
    visibility: hidden;
}}
"""

custom_css(css)

# with st.sidebar:
#     st.markdown("<center><h1><b>Verbal Vanguard</b></h1><h3>Crime & Safety Identifier</h3></center>", unsafe_allow_html=True)

def main():
    st.title("Verbal Vanguard")
    st.markdown("<h3>Crime & Safety Identifier</h3><br /><span>This prototype utilizes a BERT model to evaluate conversations and categorize them as either 'Criminal Activity' or 'Safe and Ethical'. The model is under training phase.</span><br />", unsafe_allow_html=True)

    user_text = st.text_area("Enter a conversation:- ", height=300)

    if st.button("Classify"):
        if not user_text.strip():
            st.warning("Please input a conversation to classify.")
        else:
            progress = st.progress(0)
            prediction = zero_shot_classification(user_text, progress)
            progress.empty()
            st.success(f"Predicted Category: {prediction}")

if __name__ == "__main__":
    main()