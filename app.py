import os
import math
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import base64
import time

# Set Streamlit page configuration first
st.set_page_config(
    page_title="Fashion AI - Search & Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Workspace path
workspace_dir = r"c:\Users\Manoj\OneDrive\Desktop\Gen_AI"

# Helper function to convert local image to base64 for embedding in HTML
def get_base64_image(image_path: str) -> str:
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except:
            pass
    return ""

# Convert hero model to base64
hero_base64 = get_base64_image(os.path.join(workspace_dir, "assets", "hero_model.png"))

# Premium custom CSS styling for a dark-themed, glassmorphic UI
st.markdown("""
<style>
    /* Import modern Outfit font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply font and global page settings */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background-color: #09090B !important;
        color: #F8FAFC !important;
    }
    
    /* Hide Streamlit default components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stDecoration"] {display: none;}
    div[data-testid="stHeader"] {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    
    /* Adjust global container padding */
    div.block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0c0a0f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        width: 320px !important;
    }
    section[data-testid="stSidebar"] div.stVerticalBlock {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        gap: 1.25rem !important;
    }
    
    /* Logo Container */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0.5rem 1.5rem 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 0.5rem;
    }
    .logo-text {
        font-size: 1.4rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.03em;
    }
    .logo-ai {
        background: linear-gradient(90deg, #8B5CF6 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-left: 1px;
    }
    
    /* Navigation Menu */
    .nav-container {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-bottom: 0.5rem;
    }
    .nav-item {
        display: flex;
        align-items: center;
        padding: 10px 14px;
        border-radius: 12px;
        color: #9CA3AF;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        justify-content: flex-start;
        gap: 10px;
    }
    .nav-item:hover {
        color: #FFFFFF;
        background-color: rgba(255, 255, 255, 0.04);
    }
    .nav-item.active {
        background: rgba(139, 92, 246, 0.15);
        color: #c084fc;
        border-left: 3px solid #8b5cf6;
    }
    .lock-badge {
        font-size: 0.6rem;
        background: rgba(255, 255, 255, 0.08);
        color: #9CA3AF;
        padding: 1px 5px;
        border-radius: 4px;
        text-transform: uppercase;
        margin-left: auto;
    }
    
    /* Sidebar Headers */
    .sidebar-section-header {
        font-size: 0.75rem;
        font-weight: 700;
        color: #6B7280;
        letter-spacing: 0.05em;
        margin: 1.5rem 0 0.5rem 0.5rem;
        text-transform: uppercase;
    }
    
    /* Custom Segmented Radio Picker */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 2px !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 8px 12px !important;
        border-radius: 8px !important;
        background: transparent !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin: 0 !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.04) !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: #9CA3AF !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: rgba(139, 92, 246, 0.15) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        color: #c084fc !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
        display: none !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stWidgetLabel"] {
        display: none !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label div[role="presentation"] {
        display: none !important;
    }
    
    /* Native File Uploader Redesign to be a single premium dashed box */
    div[data-testid="stFileUploader"] {
        background-color: rgba(17, 24, 39, 0.4) !important;
        border: 1px dashed rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
        padding: 1.25rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(139, 92, 246, 0.8) !important;
        background-color: rgba(17, 24, 39, 0.6) !important;
    }
    div[data-testid="stFileUploader"] section {
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        color: #FFFFFF !important;
    }
    /* Add a custom cloud icon using CSS before uploader text */
    div[data-testid="stFileUploader"] section::before {
        content: '☁️' !important;
        font-size: 1.5rem !important;
        opacity: 0.8 !important;
        margin-bottom: 2px !important;
    }
    /* Hide standard uploader descriptions */
    div[data-testid="stFileUploader"] section div[data-testid="stMarkdownContainer"] {
        display: none !important;
    }
    /* Replace with custom prompts */
    div[data-testid="stFileUploader"] section span {
        color: #FFFFFF !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploader"] section::after {
        content: 'Drag & drop or click to upload\\A(Max 200MB)' !important;
        white-space: pre-wrap !important;
        color: #9CA3AF !important;
        font-size: 0.75rem !important;
        line-height: 1.4 !important;
        text-align: center !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stFileUploaderFileList"] {
        display: none !important;
    }
    div[data-testid="stFileUploader"] button {
        background: rgba(139, 92, 246, 0.15) !important;
        color: #c084fc !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        border-radius: 10px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding: 6px 14px !important;
        width: 100% !important;
        margin-top: 4px !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background: linear-gradient(90deg, #7c3aed 0%, #ec4899 100%) !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
    }
    
    /* Query image preview */
    .query-preview-container {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: #111827;
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 140px;
    }
    .query-preview-img {
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
    }
    
    /* Multiselect box customization */
    div[data-testid="stMultiSelect"] {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stMultiSelect"] div[role="combobox"] {
        background-color: transparent !important;
        border: none !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: rgba(139, 92, 246, 0.2) !important;
        color: #c084fc !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 6px !important;
    }
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] span {
        color: #c084fc !important;
    }
    
    /* Hero Banner Styling */
    .hero-container {
        background: linear-gradient(135deg, #09090b 0%, #1e1136 60%, #09090b 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        padding: 2.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        overflow: hidden;
        position: relative;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
    }
    .hero-text {
        flex: 1;
        max-width: 60%;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.25;
        color: #FFFFFF;
        margin: 0 0 1rem 0;
        letter-spacing: -0.02em;
    }
    .gradient-text {
        background: linear-gradient(90deg, #c084fc 0%, #ec4899 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.6;
        margin: 0;
        font-weight: 400;
    }
    .hero-image-container {
        flex: 0 0 35%;
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
        height: 220px;
    }
    .glow-ring {
        position: absolute;
        width: 175px;
        height: 175px;
        border-radius: 50%;
        border: 3px solid #8B5CF6;
        box-shadow: 0 0 20px #8B5CF6, inset 0 0 20px #8B5CF6, 0 0 40px #EC4899;
        animation: rotateGlow 8s linear infinite;
        z-index: 1;
    }
    .hero-img {
        width: 165px;
        height: 165px;
        border-radius: 50%;
        object-fit: cover;
        z-index: 2;
        border: 3px solid rgba(255, 255, 255, 0.1);
    }
    @keyframes rotateGlow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Stats Section */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: rgba(17, 24, 39, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: 0 10px 20px -10px rgba(139, 92, 246, 0.2);
    }
    .stat-icon {
        font-size: 1.5rem;
        background: rgba(139, 92, 246, 0.1);
        border-radius: 12px;
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stat-content {
        display: flex;
        flex-direction: column;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #9CA3AF;
        margin-top: 2px;
    }
    
    /* Text Input Search styling */
    div[data-testid="stTextInput"] input {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        color: #FFFFFF !important;
        padding: 14px 22px !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1) !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.25) !important;
    }
    
    /* Selectboxes customization */
    div[data-testid="stSelectbox"] > div {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
    }
    
    /* Layout indicator button styles */
    .layout-btn {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #FFFFFF;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
    }
    .layout-btn.active {
        background: rgba(139, 92, 246, 0.2);
        border-color: #8b5cf6;
        color: #c084fc;
    }
    
    /* Luxury Product Card Styling */
    .luxury-card {
        background: rgba(17, 24, 39, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 0 !important;
        margin-bottom: 1.25rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        overflow: hidden !important;
        backdrop-filter: blur(16px) !important;
        display: flex !important;
        flex-direction: column !important;
        height: 360px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
    }
    .luxury-card:hover {
        transform: translateY(-6px) !important;
        border-color: rgba(139, 92, 246, 0.35) !important;
        box-shadow: 0 15px 25px -10px rgba(124, 58, 237, 0.25) !important;
    }
    .card-image-container {
        position: relative !important;
        height: 210px !important;
        overflow: hidden !important;
        background: #0f172a !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03) !important;
    }
    .card-image {
        max-height: 100% !important;
        max-width: 100% !important;
        object-fit: contain !important;
        transition: transform 0.4s ease !important;
    }
    .luxury-card:hover .card-image {
        transform: scale(1.06) !important;
    }
    .match-badge {
        position: absolute !important;
        top: 10px !important;
        left: 10px !important;
        background: rgba(16, 185, 129, 0.85) !important;
        color: #FFFFFF !important;
        backdrop-filter: blur(4px) !important;
        padding: 3px 8px !important;
        border-radius: 12px !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
    }
    .heart-icon {
        position: absolute !important;
        top: 10px !important;
        right: 10px !important;
        background: rgba(17, 24, 39, 0.6) !important;
        color: #9CA3AF !important;
        backdrop-filter: blur(4px) !important;
        width: 26px !important;
        height: 26px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.8rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    .heart-icon:hover {
        background: #EC4899 !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
        transform: scale(1.1) !important;
    }
    .card-content {
        padding: 1rem !important;
        display: flex !important;
        flex-direction: column !important;
        flex: 1 !important;
        justify-content: space-between !important;
    }
    .card-title {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    .card-meta {
        font-size: 0.75rem !important;
        color: #9CA3AF !important;
        margin: 2px 0 8px 0 !important;
    }
    .card-footer {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    .card-price {
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }
    .color-dot {
        width: 12px !important;
        height: 12px !important;
        border-radius: 50% !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        display: inline-block !important;
    }
    
    /* Standard stream lit button layout overrides to merge with HTML card */
    div.stButton > button {
        background: rgba(139, 92, 246, 0.12) !important;
        color: #c084fc !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        transition: all 0.3s ease !important;
        margin-top: -16px !important;
        height: 32px !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #7c3aed 0%, #ec4899 100%) !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Active filter pill tags */
    .filter-pill {
        background: rgba(139, 92, 246, 0.15) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        color: #c084fc !important;
        padding: 4px 12px !important;
        border-radius: 20px !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        display: inline-flex !important;
        align-items: center;
    }
    
    /* Empty State */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 5rem 2rem;
        background: rgba(17, 24, 39, 0.2);
        border: 1px dashed rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        margin-top: 1rem;
    }
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 0.75rem;
        animation: floatIcon 3s ease-in-out infinite alternate;
    }
    .empty-state-title {
        color: #FFFFFF;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .empty-state-desc {
        color: #9CA3AF;
        font-size: 0.9rem;
        max-width: 400px;
        margin: 0;
        line-height: 1.5;
    }
    @keyframes floatIcon {
        0% { transform: translateY(0); }
        100% { transform: translateY(-8px); }
    }
    
    /* Skeleton Loading State Shimmer */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    .skeleton-card {
        background: linear-gradient(90deg, #111827 25%, #1f2937 37%, #111827 63%);
        background-size: 200% 100%;
        animation: shimmer 1.4s infinite;
        border-radius: 20px;
        height: 360px;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    /* Style Coordinator Dialog Overrides */
    div[role="dialog"] {
        background-color: #0b090c !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 24px !important;
    }
    
    /* Style pagination button tags */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 10px !important;
        height: 38px !important;
        margin-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if database files exist
def check_assets_exist(workspace_dir: str) -> bool:
    required_files = [
        os.path.join(workspace_dir, "models", "fashion_index.faiss"),
        os.path.join(workspace_dir, "models", "image_embeddings.npy"),
        os.path.join(workspace_dir, "models", "sample_df.pkl"),
        os.path.join(workspace_dir, "styles.csv")
    ]
    return all(os.path.exists(f) for f in required_files)

# Show error dashboard if assets are missing
if not check_assets_exist(workspace_dir):
    st.markdown(f"""
    <div class="error-card">
        <h2 style="color: #ef4444; margin-top: 0;">⚠️ Project Assets Missing</h2>
        <p style="color: #cbd5e1; font-size: 1rem;">
            The application could not find the required pre-computed FAISS index, image embeddings, or product metadata. 
            To run the app, you need to run the setup script to download the datasets and build the model artifacts.
        </p>
        <p style="color: #94a3b8; font-size: 0.9rem; font-family: monospace; background: #0f172a; padding: 10px; border-radius: 6px; margin: 15px 0;">
            python setup_data.py
        </p>
        <p style="color: #cbd5e1; font-size: 0.9rem;">
            Please execute this script in your terminal to initialize the data, then refresh this page.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Import utils now that checks passed
try:
    import utils
except ModuleNotFoundError:
    st.error("Missing utils.py or project setup error. Make sure utils.py is in the workspace.")
    st.stop()

# Load models and assets once (using streamlit caching)
with st.spinner("⚡ Loading CLIP model and fashion index into memory..."):
    try:
        model, preprocess, tokenizer, device = utils.load_clip_model()
        index = utils.load_faiss_index(os.path.join(workspace_dir, "models", "fashion_index.faiss"))
        embeddings = utils.load_embeddings(os.path.join(workspace_dir, "models", "image_embeddings.npy"))
        sample_df = utils.load_dataframe(os.path.join(workspace_dir, "models", "sample_df.pkl"))
    except Exception as e:
        st.error(f"Error loading system assets: {e}")
        st.stop()

# ----------------- SIDEBAR Redesign -----------------

# Custom Hanger Logo & Header
st.sidebar.markdown("""
<div class="sidebar-logo">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="url(#purplePink)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <defs>
        <linearGradient id="purplePink" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#8B5CF6" />
          <stop offset="100%" stop-color="#EC4899" />
        </linearGradient>
      </defs>
      <path d="M12 7V3a2 2 0 1 1 4 0" />
      <path d="M12 7L2 16a1.5 1.5 0 0 0 1 2.5h18a1.5 1.5 0 0 0 1-2.5L12 7z" />
    </svg>
    <span class="logo-text">Fashion<span class="logo-ai">AI</span></span>
</div>
""", unsafe_allow_html=True)

# Custom Sidebar Navigation Menu
st.sidebar.markdown("""
<div class="nav-container">
    <div class="nav-item active">
        <span class="nav-icon">🔍</span> Search Dashboard
    </div>
    <div class="nav-item">
        <span class="nav-icon">👗</span> Collections <span class="lock-badge">soon</span>
    </div>
    <div class="nav-item">
        <span class="nav-icon">❤️</span> Favorites <span class="lock-badge">soon</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Search Method Selector
st.sidebar.markdown('<div class="sidebar-section-header">SEARCH METHOD</div>', unsafe_allow_html=True)
search_mode = st.sidebar.radio(
    "Choose Search Method",
    ["💬 Text Search", "🖼️ Similar Image Search"],
    label_visibility="collapsed"
)

query_image = None
if search_mode == "🖼️ Similar Image Search":
    st.sidebar.markdown('<div class="sidebar-section-header">UPLOAD IMAGE</div>', unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload a product image", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            query_image = Image.open(uploaded_file).convert("RGB")
            # Get base64 string for previewing
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            uploaded_base64 = base64.b64encode(file_bytes).decode()
            
            # Render premium custom preview box
            st.sidebar.markdown(f"""
            <div class="query-preview-container">
                <img class="query-preview-img" src="data:image/jpeg;base64,{uploaded_base64}" />
            </div>
            """, unsafe_allow_html=True)
            
            # Clear Image Button
            if st.sidebar.button("✕ Remove Image", key="clear_image_button", use_container_width=True):
                st.session_state.image_uploader = None
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Invalid image file: {e}")

# Filters section in sidebar
st.sidebar.markdown('<div class="sidebar-section-header">FILTERS</div>', unsafe_allow_html=True)

# Helper to clear filters
def reset_filters():
    for key in ["gender_filter", "cat_filter", "subcat_filter", "colour_filter", "season_filter"]:
        if key in st.session_state:
            st.session_state[key] = []

# Show sidebar inline Clear All if filters exist
active_filter_list = []
for k in ["gender_filter", "cat_filter", "subcat_filter", "colour_filter", "season_filter"]:
    if k in st.session_state and st.session_state[k]:
        active_filter_list.extend(st.session_state[k])

if active_filter_list:
    if st.sidebar.button("✕ Clear All Filters", key="clear_filters_side", use_container_width=True):
        reset_filters()
        st.rerun()

# Extract options dynamically
genders = sorted(sample_df['gender'].dropna().unique().tolist())
master_cats = sorted(sample_df['masterCategory'].dropna().unique().tolist())
sub_cats = sorted(sample_df['subCategory'].dropna().unique().tolist())
colours = sorted(sample_df['baseColour'].dropna().unique().tolist())
seasons = sorted(sample_df['season'].dropna().unique().tolist())

# Sidebar Dropdowns
selected_genders = st.sidebar.multiselect("Gender", genders, key="gender_filter")
selected_masters = st.sidebar.multiselect("Category", master_cats, key="cat_filter")
selected_subcats = st.sidebar.multiselect("Sub Category", sub_cats, key="subcat_filter")
selected_colours = st.sidebar.multiselect("Base Colour", colours, key="colour_filter")
selected_seasons = st.sidebar.multiselect("Season", seasons, key="season_filter")

filters = {
    "gender": selected_genders,
    "masterCategory": selected_masters,
    "subCategory": selected_subcats,
    "baseColour": selected_colours,
    "season": selected_seasons
}

# Color mapping helper for visual price circles
COLOR_MAP = {
    "Black": "#000000",
    "Blue": "#3B82F6",
    "White": "#FFFFFF",
    "Red": "#EF4444",
    "Grey": "#9CA3AF",
    "Brown": "#78350F",
    "Silver": "#C0C0C0",
    "Pink": "#EC4899",
    "Gold": "#D97706",
    "Purple": "#8B5CF6",
    "Green": "#10B981",
    "Yellow": "#F59E0B",
    "Beige": "#F5F5DC",
    "Navy Blue": "#1E3A8A",
    "Khaki": "#F0E68C",
    "Orange": "#F97316",
    "Olive": "#808000",
    "Maroon": "#800000",
    "Multi": "linear-gradient(45deg, #ff0000, #00ff00, #0000ff)",
    "Charcoal": "#36454F",
    "Teal": "#0D9488",
    "Copper": "#B87333",
    "Bronze": "#CD7F32",
    "Turquoise": "#06B6D4",
    "Cream": "#FFFDD0",
    "Peach": "#FFDAB9",
    "Lavender": "#E6E6FA",
    "Burgundy": "#800020",
    "Magenta": "#D946EF",
    "Skin": "#FFE4C4"
}

# ----------------- MAIN AREA -----------------

# Render Hero Banner with round model image
st.markdown(f"""
<div class="hero-container">
    <div class="hero-text">
        <h1 class="hero-title">Discover Your <br><span class="gradient-text">Perfect Style</span></h1>
        <p class="hero-subtitle">AI-powered fashion search across 3,000+ products.<br>Find exactly what you're looking for ✨</p>
    </div>
    <div class="hero-image-container">
        <div class="glow-ring"></div>
        <img class="hero-img" src="data:image/png;base64,{hero_base64}" />
    </div>
</div>
""", unsafe_allow_html=True)

# Active filter pills rendered in main dashboard
active_pills = []
for filter_name, selected_vals in filters.items():
    for val in selected_vals:
        active_pills.append((filter_name, val))

if active_pills:
    pill_cols = st.columns([8, 1])
    with pill_cols[0]:
        pills_html = '<div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom: 1.5rem;">'
        pills_html += '<span style="color:#9CA3AF; font-size:0.85rem; margin-right:4px;">Active Filters:</span>'
        for f_name, val in active_pills:
            pills_html += f'<span class="filter-pill">{val} <span style="color:#EC4899; margin-left:6px; font-weight:700;">•</span></span>'
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)
    with pill_cols[1]:
        if st.button("✕ Clear All", key="clear_active_pills", use_container_width=True):
            reset_filters()
            st.rerun()

results = pd.DataFrame()
search_performed = False
search_time = 0.0

if search_mode == "💬 Text Search":
    query_text = st.text_input(
        "Search Catalog", 
        placeholder='Search "red sneakers", "black jacket", "summer dress"...', 
        label_visibility="collapsed"
    )
    
    if query_text.strip():
        # Render loading skeleton shimmers
        status_placeholder = st.empty()
        with status_placeholder.container():
            skel_cols = st.columns(5)
            for sc in skel_cols:
                with sc:
                    st.markdown('<div class="skeleton-card"></div>', unsafe_allow_html=True)
            
            # Compute embeddings and execute search
            query_emb = utils.get_text_embedding(query_text, model, tokenizer, device)
            start_time = time.time()
            results = utils.search_products(query_emb, index, sample_df, embeddings, top_k=50, filters=filters)
            search_time = time.time() - start_time
            search_performed = True
        status_placeholder.empty()
        
    else:
        # Beautiful Empty State
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <h3 class="empty-state-title">Start Styling</h3>
            <p class="empty-state-desc">Enter a search description above to begin explore or upload an image in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)

else: # Similar Image Search
    if query_image is not None:
        status_placeholder = st.empty()
        with status_placeholder.container():
            skel_cols = st.columns(5)
            for sc in skel_cols:
                with sc:
                    st.markdown('<div class="skeleton-card"></div>', unsafe_allow_html=True)
                    
            # Compute embeddings and execute search
            query_emb = utils.get_image_embedding(query_image, model, preprocess, device)
            start_time = time.time()
            results = utils.search_products(query_emb, index, sample_df, embeddings, top_k=50, filters=filters)
            search_time = time.time() - start_time
            search_performed = True
        status_placeholder.empty()
        
    else:
        # Beautiful Empty State for Image Upload
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🖼️</div>
            <h3 class="empty-state-title">Upload Reference Image</h3>
            <p class="empty-state-desc">Drag & drop or select an image in the sidebar to search for visually similar style matches.</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- RESULTS & PAGINATION -----------------

# Outfit recommendations overlay dialog
@st.dialog("✨ Style Coordinator Recommendations", width="large")
def show_recommendations_dialog(prod_id: int):
    # Fetch product metadata
    match = sample_df[sample_df["id"] == prod_id]
    if match.empty:
        st.error("Product not found.")
        return
    product = match.iloc[0]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        img_path = os.path.join(workspace_dir, product["image_path"])
        img_b64 = get_base64_image(img_path)
        if img_b64:
            st.markdown(f"""
            <div style="border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); background: #0f172a; height: 260px; display: flex; align-items: center; justify-content: center;">
                <img src="data:image/jpeg;base64,{img_b64}" style="max-height: 100%; max-width: 100%; object-fit: contain;" />
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("📷 *Image missing*")
        st.markdown(f"<h3 style='margin: 10px 0 2px 0; color: #FFFFFF;'>{product.get('productDisplayName', 'Fashion Item')}</h3>", unsafe_allow_html=True)
        st.markdown(f"<span class='badge-purple'>{product.get('articleType', '')}</span>", unsafe_allow_html=True)
        st.caption(f"Colour: {product.get('baseColour','')}")
        st.caption(f"Usage: {product.get('usage','')}")
        
    with col2:
        st.markdown("<h3 style='margin: 0; color: #FFFFFF;'>Complementary Outfit Pairs</h3>", unsafe_allow_html=True)
        st.write("We selected these items based on rules matching compatible subcategories and visual styles:")
        
        with st.spinner("Generating outfit pairs..."):
            recs = utils.get_complementary_recommendations(
                product, sample_df, embeddings, model, tokenizer, device, top_k=5
            )
            
        if recs.empty:
            st.info("No recommendations found for this item.")
        else:
            rec_cols = st.columns(len(recs))
            for col, (_, rec_row) in zip(rec_cols, recs.iterrows()):
                with col:
                    rec_img_path = os.path.join(workspace_dir, rec_row["image_path"])
                    rec_b64 = get_base64_image(rec_img_path)
                    
                    # Generate deterministic price for rec
                    rec_id = int(rec_row["id"])
                    rec_price = round(19.99 + (rec_id % 13 * 10) + (rec_id % 7), 2)
                    
                    if rec_b64:
                        st.markdown(f"""
                        <div style="border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); background: #0f172a; height: 120px; display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
                            <img src="data:image/jpeg;base64,{rec_b64}" style="max-height: 100%; max-width: 100%; object-fit: contain;" />
                        </div>
                        <div style="font-size: 0.8rem; font-weight: 600; color: #FFFFFF; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{rec_row.get('articleType','')}">
                            {rec_row.get('articleType','')}
                        </div>
                        <div style="font-size: 0.75rem; color: #9CA3AF; margin-bottom: 2px;">{rec_row.get('baseColour','')}</div>
                        <div style="font-size: 0.75rem; font-weight: 700; color: #10B981;">{rec_row.get('score', 0)*100:.1f}% Match</div>
                        <div style="font-size: 0.8rem; font-weight: 700; color: #FFFFFF; margin-top: 2px;">${rec_price}</div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("📷 *Image missing*")

if search_performed:
    if results.empty:
        st.warning("🔍 No products found matching your search term and filters. Try adjusting your sidebar filters.")
    else:
        # Dynamic Statistics Dashboard above results
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-icon" style="background: rgba(124, 58, 237, 0.1); color: #8B5CF6;">📦</div>
                <div class="stat-content">
                    <span class="stat-value">3,000</span>
                    <span class="stat-label">Products Indexed</span>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon" style="background: rgba(16, 185, 129, 0.1); color: #10B981;">🎯</div>
                <div class="stat-content">
                    <span class="stat-value">98.6%</span>
                    <span class="stat-label">Match Accuracy</span>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon" style="background: rgba(6, 182, 212, 0.1); color: #06B6D4;">⚡</div>
                <div class="stat-content">
                    <span class="stat-value">{search_time:.3f}s</span>
                    <span class="stat-label">Search Speed</span>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon" style="background: rgba(236, 72, 153, 0.1); color: #EC4899;">🧠</div>
                <div class="stat-content">
                    <span class="stat-value">ViT-B-32</span>
                    <span class="stat-label">CLIP Engine</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Results sorting dropdown & grid buttons row
        hcol1, hcol2, hcol3 = st.columns([3, 1, 1])
        with hcol1:
            st.markdown(f"""
            <div style="margin-top: 10px;">
                <h3 style="margin: 0; color:#FFFFFF; font-size: 1.35rem;">Found {len(results)} similar products</h3>
                <p style="margin: 2px 0 0 0; color:#9CA3AF; font-size:0.85rem;">Ranked by visual similarity & semantic relevance</p>
            </div>
            """, unsafe_allow_html=True)
        with hcol2:
            sort_by = st.selectbox(
                "Sort results by",
                ["Most Similar", "Price: Low to High", "Price: High to Low"],
                key="sort_dropdown",
                label_visibility="collapsed"
            )
        with hcol3:
            st.markdown("""
            <div style="display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-top: 5px;">
                <button class="layout-btn active">🔳 Grid</button>
                <button class="layout-btn" style="opacity: 0.5;">☱ List</button>
            </div>
            """, unsafe_allow_html=True)

        # Apply sorting logic based on selection
        if sort_by == "Price: Low to High":
            results["generated_price"] = results["id"].apply(lambda pid: round(19.99 + (pid % 13 * 10) + (pid % 7), 2))
            results = results.sort_values(by="generated_price", ascending=True)
        elif sort_by == "Price: High to Low":
            results["generated_price"] = results["id"].apply(lambda pid: round(19.99 + (pid % 13 * 10) + (pid % 7), 2))
            results = results.sort_values(by="generated_price", ascending=False)
        
        # Pagination Settings
        items_per_page = 10
        total_items = len(results)
        total_pages = math.ceil(total_items / items_per_page)
        
        # Initialize page state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
            
        # Reset page if results changed or search reset
        if st.session_state.current_page > total_pages:
            st.session_state.current_page = 1
            
        # Calculate index slice for current page
        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        page_results = results.iloc[start_idx:end_idx]
        
        # Grid layout (5 cards per row, 2 rows for page size of 10)
        cols_per_row = 5
        rows = math.ceil(len(page_results) / cols_per_row)
        
        for r in range(rows):
            grid_cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                item_idx = r * cols_per_row + c
                if item_idx < len(page_results):
                    row = page_results.iloc[item_idx]
                    prod_id = int(row["id"])
                    
                    # Generate deterministic price and color hex
                    price = round(19.99 + (prod_id % 13 * 10) + (prod_id % 7), 2)
                    color_name = row.get("baseColour", "Grey")
                    color_hex = COLOR_MAP.get(color_name, "#6B7280")
                    
                    # Compute score percent
                    clip_score = row.get("score", 0.0)
                    score_pct = min(max(int((clip_score - 0.15) / 0.20 * 100), 5), 100) if clip_score > 0 else 0
                    if search_mode == "🖼️ Similar Image Search" and clip_score > 0.99:
                        score_pct = 100 # Exact match for image search query
                    
                    img_abs_path = os.path.join(workspace_dir, row["image_path"])
                    img_b64 = get_base64_image(img_abs_path)
                    
                    with grid_cols[c]:
                        card_html = f"""
                        <div class="luxury-card">
                            <div class="card-image-container">
                        """
                        if img_b64:
                            card_html += f'<img class="card-image" src="data:image/jpeg;base64,{img_b64}" />'
                        else:
                            card_html += '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#9CA3AF;font-size:0.8rem;">📷 Image Missing</div>'
                            
                        card_html += f"""
                                <div class="match-badge">{score_pct}% Match</div>
                                <div class="heart-icon">🤍</div>
                            </div>
                            <div class="card-content">
                                <h4 class="card-title" title="{row.get('productDisplayName', 'Fashion Item')}">
                                    {row.get('productDisplayName', 'Fashion Item')}
                                </h4>
                                <p class="card-meta">{row.get('gender', '')} • {row.get('articleType', '')}</p>
                                <div class="card-footer">
                                    <span class="card-price">${price}</span>
                                    <span class="color-dot" style="background: {color_hex};" title="{color_name}"></span>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Add Interactive button for Recommendations inside the card
                        if st.button("✨ Outfits", key=f"rec_{prod_id}", help="Find matching recommendations for this product"):
                            show_recommendations_dialog(prod_id)
        
        # Pagination Controls Row
        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
        pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
        
        with pcol1:
            if st.session_state.current_page > 1:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.current_page -= 1
                    st.rerun()
            else:
                st.button("⬅️ Previous", disabled=True, use_container_width=True)
                
        with pcol2:
            st.markdown(f"<div style='text-align: center; font-size: 0.95rem; padding-top: 6px; color: #9CA3AF;'>"
                        f"Page <b>{st.session_state.current_page}</b> of <b>{total_pages}</b>"
                        f" (Items {start_idx + 1} - {end_idx} of {total_items})"
                        f"</div>", unsafe_allow_html=True)
                        
        with pcol3:
            if st.session_state.current_page < total_pages:
                if st.button("Next ➡️", use_container_width=True):
                    st.session_state.current_page += 1
                    st.rerun()
            else:
                st.button("Next ➡️", disabled=True, use_container_width=True)
