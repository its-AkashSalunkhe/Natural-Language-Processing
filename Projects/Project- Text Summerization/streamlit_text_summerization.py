import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# 1. Page Configuration
st.set_page_config(
    page_title="Text Summarizer App",
    page_icon="📝",
    layout="centered"
)

# 2. Load SpaCy Model (Cached for performance)
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

with st.spinner("Loading NLP model... Please wait."):
    nlp = load_model()

# 3. Summarization Function
def summarize_text(text, summary_ratio=0.4):
    doc = nlp(text)
    
    # Setup stopwords and punctuation
    stopwords = list(STOP_WORDS)
    puncts = punctuation
    
    # Calculate word frequencies
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in puncts:
            if word.text not in word_frequencies:
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
                
    if not word_frequencies:
        return []

    max_frequency = max(word_frequencies.values())
    
    # Normalize frequencies
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency
        
    # Sentence Tokenization
    sentence_tokens = list(doc.sents)
    if not sentence_tokens:
        return []

    # Calculate sentence scores
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
                    
    # Select top sentences
    select_length = max(1, int(len(sentence_tokens) * summary_ratio))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    
    final_summary = [word.text for word in summary]
    return final_summary

# 4. Streamlit UI Layout
st.title("📝 NLP Text Summarization App")
st.write("An extractive text summarization tool powered by **SpaCy** and built with **Streamlit**.")

# Sidebar controls
st.sidebar.header("Settings")
summary_ratio = st.sidebar.slider(
    "Summary Size Ratio", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.4, 
    step=0.05,
    help="Determines what fraction of the original sentences to keep."
)

# Text Input Area
text_input = st.text_area(
    "Enter text to summarize:", 
    height=250,
    placeholder="Paste your long article or paragraph here..."
)

# Summarize Button
if st.button("Generate Summary", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text to summarize!")
    else:
        with st.spinner("Analyzing and summarizing text..."):
            summary_sentences = summarize_text(text_input, summary_ratio)
            
            if summary_sentences:
                st.subheader("Summary Output")
                combined_summary = " ".join(summary_sentences)
                st.success(combined_summary)
                
                with st.expander("View Sentence-by-Sentence Breakdown"):
                    for i, sent in enumerate(summary_sentences, 1):
                        st.markdown(f"**{i}.** {sent}")
            else:
                st.error("Could not generate a summary. Try providing longer text.")