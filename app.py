import streamlit as st
from predict import predict_news

st.set_page_config(page_title="AI Misinformation Detector")

st.title("AI Misinformation Detector")

text = st.text_area("Enter news text")

if st.button("Check"):
    if text.strip():
        label, confidence = predict_news(text)

        st.subheader("Prediction")
        st.write(label)

        st.subheader("Confidence")
        st.write(round(confidence, 4))