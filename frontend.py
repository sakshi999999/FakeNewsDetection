import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from backend import predict_news
import time

# Set page configuration with a wider layout
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .fake-label {
        color: #DC2626;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .real-label {
        color: #059669;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .uncertain-label {
        color: #D97706;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .info-text {
        color: #4B5563;
        font-size: 0.9rem;
    }
    .highlight {
        background-color: #FEF3C7;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.3rem;
        height: 2.5rem;
        font-weight: 500;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .fake-box {
        background-color: rgba(220, 38, 38, 0.1);
        border-left: 5px solid #DC2626;
    }
    .real-box {
        background-color: rgba(5, 150, 105, 0.1);
        border-left: 5px solid #059669;
    }
    .uncertain-box {
        background-color: rgba(217, 119, 6, 0.1);
        border-left: 5px solid #D97706;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .progress-container {
        width: 100%;
        background-color: #E5E7EB;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .progress-bar {
        height: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        color: white;
        font-weight: 600;
    }
    .progress-fake {
        background-color: #DC2626;
    }
    .progress-real {
        background-color: #059669;
    }
    .input-container {
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .example-container {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-text">
This advanced tool uses machine learning with TF-IDF to analyze news content and determine if it's likely to be real or fake.
The model examines patterns in the text that are commonly associated with misinformation.
</div>
""", unsafe_allow_html=True)

# Sidebar for settings and information
with st.sidebar:
    st.markdown('<div class="sub-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <p>This tool uses machine learning to analyze news content and determine if it's likely to be real or fake news.</p>
    <p>The model has been trained on thousands of articles and looks for patterns in the text that are commonly associated with misinformation.</p>
    <p><strong>Note</strong>: No AI system is perfect. Always verify news from multiple reliable sources.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">🔍 How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <ol>
        <li>Paste your news text in the input box</li>
        <li>Click "Analyze Text"</li>
        <li>Review the prediction and confidence scores</li>
        <li>Check the detailed analysis for more insights</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key input is removed since we're not using NewsAPI anymore

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="sub-header">📝 Text Analysis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Text input
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown("<p><strong>Paste news content to analyze:</strong></p>", unsafe_allow_html=True)
    user_input = st.text_area("", height=150, label_visibility="collapsed", 
                              placeholder="Paste news article text here...")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Example buttons with better styling
    st.markdown('<div class="example-container">', unsafe_allow_html=True)
    st.markdown("<p><strong>Or try an example:</strong></p>", unsafe_allow_html=True)
    example_col1, example_col2 = st.columns(2)
    with example_col1:
        if st.button("📌 Example Real News", use_container_width=True):
            user_input = """Scientists have discovered a new species of deep-sea fish that can withstand extreme pressure. The findings, published today in the journal Nature, suggest adaptations that could have implications for medical research on high blood pressure. The research team spent three years studying the fish at depths of over 8,000 meters."""
            st.session_state.user_input = user_input
            
    with example_col2:
        if st.button("📌 Example Fake News", use_container_width=True):
            user_input = """BREAKING: Scientists discover fish that can speak 5 human languages! The revolutionary fish, found in the Atlantic Ocean, can communicate in English, Spanish, French, German and Russian. Government officials are keeping the fish in a secret location. Experts say this could change human-animal communication forever."""
            st.session_state.user_input = user_input
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sub-header">🧠 Quick Tips</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <p><strong>Signs of potential fake news:</strong></p>
    <ul>
        <li>Sensational or clickbait headlines</li>
        <li>Emotional language and exaggerations</li>
        <li>Lack of sources or citations</li>
        <li>Unusual website URLs or author names</li>
        <li>Poor grammar and spelling</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# 🔍 Prediction
if analyze_button and user_input:
    # Create a placeholder for the spinner
    spinner_placeholder = st.empty()
    
    with spinner_placeholder.container():
        with st.spinner("Analyzing text..."):
            # Add a small delay to show the spinner (for demo purposes)
            time.sleep(0.5)
            
            # Get prediction
            result = predict_news(user_input)
            
            if len(result) == 3:  # If we're using the updated backend
                label, confidence, debug_info = result
            else:
                label, confidence = result
                debug_info = {"fake_conf": 0, "real_conf": 0}
    
    # Clear the spinner
    spinner_placeholder.empty()
    
    # Display results in a nice layout
    results_col1, results_col2 = st.columns([3, 2])
    
    with results_col1:
        st.markdown('<div class="sub-header">🔍 Analysis Results</div>', unsafe_allow_html=True)
        
        # Determine the box style based on the prediction
        box_class = ""
        if "FAKE" in label:
            box_class = "fake-box"
            label_class = "fake-label"
            icon = "⚠️"
        elif "REAL" in label:
            box_class = "real-box"
            label_class = "real-label"
            icon = "✅"
        elif "UNCERTAIN" in label:
            box_class = "uncertain-box"
            label_class = "uncertain-label"
            icon = "❓"
        else:
            box_class = "uncertain-box"
            label_class = "uncertain-label"
            icon = "ℹ️"
        
        # Display the prediction box
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <div class="{label_class}">{icon} {label}</div>
            <p>Confidence: {confidence:.1f}%</p>
            <div class="info-text">
                Based on analysis of text patterns, language, and content structure.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display confidence bars
        if debug_info:
            st.markdown("<p><strong>Confidence Breakdown:</strong></p>", unsafe_allow_html=True)
            
            # Fake news confidence bar
            fake_conf = debug_info['fake_conf']
            st.markdown(f"""
            <div>FAKE: {fake_conf:.1f}%</div>
            <div class="progress-container">
                <div class="progress-bar progress-fake" style="width:{fake_conf}%"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Real news confidence bar
            real_conf = debug_info['real_conf']
            st.markdown(f"""
            <div>REAL: {real_conf:.1f}%</div>
            <div class="progress-container">
                <div class="progress-bar progress-real" style="width:{real_conf}%"></div>
            </div>
            """, unsafe_allow_html=True)
    
    with results_col2:
        st.markdown('<div class="sub-header">📊 Text Analysis</div>', unsafe_allow_html=True)
        
        # Text statistics card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Display text length and statistics
        text_length = len(user_input.split())
        sentence_count = len([s for s in user_input.split('.') if s.strip()])
        
        st.markdown(f"<p><strong>Text Length:</strong> {text_length} words</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Sentences:</strong> {sentence_count}</p>", unsafe_allow_html=True)
        
        # Average sentence length
        avg_sentence_length = text_length / max(1, sentence_count)
        st.markdown(f"<p><strong>Avg. Sentence Length:</strong> {avg_sentence_length:.1f} words</p>", unsafe_allow_html=True)
        
        # Display a warning if the text is too short
        if text_length < 20:
            st.markdown("""
            <div style="color: #DC2626; margin-top: 0.5rem;">
                ⚠️ Text is very short, which might affect prediction accuracy.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a feedback section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<p><strong>Was this prediction helpful?</strong></p>", unsafe_allow_html=True)
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            st.button("👍 Yes", use_container_width=True)
        with feedback_col2:
            st.button("👎 No", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display text excerpt with highlights (if text is long enough)
    if text_length > 50:
        st.markdown('<div class="sub-header">📝 Text Excerpt</div>', unsafe_allow_html=True)
        
        # Get a sample of the text (first 200 characters)
        excerpt = user_input[:200] + "..." if len(user_input) > 200 else user_input
        
        st.markdown(f"""
        <div class="card">
            <p>{excerpt}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2023 Fake News Detector | Powered by Machine Learning | This tool is for educational purposes only.</p>
    <p>Always verify information from multiple reliable sources before drawing conclusions.</p>
</div>
""", unsafe_allow_html=True)
