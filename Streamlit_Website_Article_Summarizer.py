# streamlit run Streamlit_Website_Article_Summarizer.py

import streamlit as st
from streamlit import logger
from crewai_tools import ScrapeWebsiteTool
from langchain_groq import ChatGroq
from groq import Groq
import os


import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

app_logger = logger.get_logger('SMI_APP')
app_logger.info(f"version sqlite : {sqlite3.sqlite_version}")
app_logger.info(f"version sys : {sys.version}")



# Streamlit App Configuration
st.set_page_config(page_title="Article Summarizer & Translator", page_icon="📄")

# Initialize Groq client (move to global scope)
API_KEY = "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"
os.environ["GROQ_API_KEY"] = API_KEY
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Global State Variables ---
if "extracted_article" not in st.session_state:
    st.session_state.extracted_article = None

# --- Helper Functions ---
def text_extractor_agent(model_input,website_content,URL_PATH):
    prompt = f"""
        You are a agent that is specialized in extracting journalist article from content that has been\
        scraped from a website page. You know that the content you receive contains a lot of unnessary information \
        but you are able to focus your attention specifically on the journalist article. You also know that within \
        the URL itself, often time there is information about the topic of the article you need to extract so you can \
        use the URL link to give you some indication of what section of the content you need to extract. \
        
        Expected output: the entire article exactly how it was written by the journalist, no summary, only the entire article word for word.
        
        Here's the URL where the content has been extracted: {URL_PATH} \

        Here's the website content from where to extract the journalist article: {website_content}
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_input,
        temperature=0,
        max_tokens=32000,
        top_p=1
    )
    return response.choices[0].message.content


def text_summarizer_agent(model_input,content_input,output_language):
    prompt = f"""
        You are a agent that is specialized in summarizing long piece of text and known to identify the most critical information of journalistic article such as: \
        Main arguments or findings, Key statistics or data, Quotes or statements from experts. \
        and provide a quick run down of what the piece of text was about. \
        
        Output expectations: \
        -A well-structured summary that accurately captures the article's main points written in {output_language} and using this language written characters. \
        -Clear and concise language, free of jargon and technical terms unless specified. \
        -Proper formatting, including headings, bullet points, and white space for readability. \
        
        Rules you need to follow: \
        -Do not make up facts, stay on the exact content and information provided to you, nothing more. \
        -Do not include your own biais or interpretation of the facts and informations you have to summarize. \
        -Always structure your output in 3 parts that you ouput the names in {output_language}: 1) Abstract, 2) Executive summary and 3) Bullet Points Summary \
        
        Here's the piece of text you need to summarize: {content_input}
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_input,
        temperature=0,
        max_tokens=5000,
        top_p=1
    )
    return response.choices[0].message.content



# --- Streamlit UI ---

# Header
st.title("📰 Article Summarizer & Translator")
st.write("Extract, summarize, and translate news articles with ease!")

# Input Area
model_selected = st.selectbox("Model", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"])  
user_language = st.selectbox("Summary Language", ["English", "French", "Spanish", "German", "Italian", "Portuguese", "Filipino", "Japanese", "Korean",  "Arabic", "Hindi", "Russian"]) 

website_link = st.text_input("Paste the article URL:")

if st.button("Process Article"):
    with st.spinner("Processing article..."):
        if st.session_state.extracted_article is None or website_link != st.session_state.last_url:  # Check if re-extraction is needed
            # Extract article
            website_scrape_tool = ScrapeWebsiteTool(website_url=website_link)
            website_text_content = website_scrape_tool.run()
            st.session_state.extracted_article = text_extractor_agent(model_selected, website_text_content, website_link)

            # Store the URL for comparison on the next run
            st.session_state.last_url = website_link

        # Summarize article (always re-summarize based on language)
        response_summary_agent = text_summarizer_agent(model_selected, st.session_state.extracted_article, user_language)

    # Display Results
    st.subheader("Extracted Article:")
    st.write(st.session_state.extracted_article)

    st.subheader(f"Summarized Article ({user_language}):")
    st.code(response_summary_agent, language="text")
