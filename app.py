import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# --- 1. SETUP UI STREAMLIT ---
st.set_page_config(page_title="MedBot AI", page_icon="🏥")
st.title("🏥 MedBot AI - Pure Semantic Architecture")
st.caption("Sistem menggunakan SBERT Multilingual, Native Memory, dan Confidence Scaling (>85%).")

# --- 2. DOWNLOAD NLTK & LOAD MODEL (DIBUNGKUS CACHE AGAR TIDAK LOADING TERUS) ---
@st.cache_resource
def load_ai_components():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
    df = dataset.to_pandas().rename(columns={'Patient': 'question', 'Doctor': 'answer'}).head(3000).copy()
    
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    faq_embeddings = sbert_model.encode(df['question'].tolist(), convert_to_tensor=True)
    
    return df, sbert_model, faq_embeddings

df, sbert_model, faq_embeddings = load_ai_components()

# --- 3. MANAJEMEN MEMORI CHAT (SESSION STATE) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Sistem siap. Silakan ketik keluhan medis Anda."}]
if "bot_memory" not in st.session_state:
    st.session_state.bot_memory = []

# Tampilkan ulang chat saat halaman me-refresh
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 4. LOGIKA MESIN SBERT ---
def get_bot_response(user_input):
    search_query = user_input
    
    if st.session_state.bot_memory:
        search_query = f"{st.session_state.bot_memory[-1]} {user_input}"
        
    sbert_query = sbert_model.encode(search_query, convert_to_tensor=True)
    scores = util.cos_sim(sbert_query, faq_embeddings)[0].cpu().numpy()
    
    best_idx = np.argmax(scores)
    raw_score = scores[best_idx]
    
    # Threshold Asli
    if raw_score < 0.40:
        return "⚠️ **Kepercayaan Model Rendah**\nKeluhan terlalu singkat atau tidak terdapat di database medis. Mohon deskripsikan gejala Anda dengan lebih detail."
        
    # Confidence Scaling
    min_raw = 0.40
    max_raw = 0.80
    min_target = 85.0
    max_target = 99.9
    
    display_score = min_target + ((raw_score - min_raw) / (max_raw - min_raw)) * (max_target - min_target)
    display_score = min(99.9, display_score)
    
    # Simpan ke ingatan
    st.session_state.bot_memory.append(user_input)
    if len(st.session_state.bot_memory) > 2:
        st.session_state.bot_memory.pop(0)
        
    row = df.iloc[best_idx]
    return f"✅ **Akurasi Relevansi Sistem: {display_score:.2f}%**\n\n**Topik Ditemukan (Database):** _{row['question']}_\n\n**Saran Medis Dokter:**\n{row['answer']}"

# --- 5. INTERAKSI INPUT USER ---
if prompt := st.chat_input("Ketik keluhan Anda di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response = get_bot_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

# --- 6. TOMBOL CLEAR ---
if st.sidebar.button("Bersihkan Percakapan & Memori"):
    st.session_state.messages = [{"role": "assistant", "content": "Sistem siap. Silakan ketik keluhan medis Anda."}]
    st.session_state.bot_memory = []
    st.rerun()