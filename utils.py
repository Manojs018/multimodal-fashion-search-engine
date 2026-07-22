import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import open_clip
import faiss
from PIL import Image
import streamlit as st

@st.cache_resource
def load_clip_model() -> Tuple[Any, Any, Any, str]:
    """
    Loads the OpenCLIP ViT-B-32 model and its preprocessors.
    Caches the model to prevent reloading on every run.
    
    Returns:
        Tuple[model, preprocess, tokenizer, device]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using the standard ViT-B-32 model trained on openai dataset
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device

@st.cache_resource
def load_faiss_index(index_path: str) -> faiss.IndexFlatIP:
    """
    Loads and caches the FAISS vector index.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at: {index_path}")
    return faiss.read_index(index_path)

@st.cache_data
def load_embeddings(emb_path: str) -> np.ndarray:
    """
    Loads and caches the saved image embeddings.
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings file not found at: {emb_path}")
    return np.load(emb_path)

@st.cache_data
def load_dataframe(df_path: str) -> pd.DataFrame:
    """
    Loads and caches the pickled sample dataframe containing metadata.
    """
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Metadata pickle not found at: {df_path}")
    return pd.read_pickle(df_path)

def get_text_embedding(query: str, model: Any, tokenizer: Any, device: str) -> np.ndarray:
    """
    Generates a normalized text embedding using OpenCLIP.
    
    Args:
        query: Search query string.
        model: Loaded CLIP model.
        tokenizer: CLIP tokenizer.
        device: 'cuda' or 'cpu'.
        
    Returns:
        np.ndarray: Normalized text embedding array.
    """
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.cpu().numpy().astype('float32')

def get_image_embedding(image: Image.Image, model: Any, preprocess: Any, device: str) -> np.ndarray:
    """
    Generates a normalized image embedding for a PIL image using OpenCLIP.
    
    Args:
        image: PIL Image object.
        model: Loaded CLIP model.
        preprocess: CLIP image preprocessor.
        device: 'cuda' or 'cpu'.
        
    Returns:
        np.ndarray: Normalized image embedding array.
    """
    img_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype('float32')

def search_products(
    query_embedding: np.ndarray,
    index: faiss.IndexFlatIP,
    sample_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Searches products matching the query embedding.
    Uses FAISS directly for unfiltered queries, and numpy vector operations 
    on filtered DataFrame subsets to guarantee exact filter matches.
    
    Args:
        query_embedding: Search query embedding of shape (1, 512) or (512,).
        index: Loaded FAISS index.
        sample_df: Loaded sample products dataframe.
        embeddings: Loaded image embeddings matrix (N, 512).
        top_k: Number of products to return.
        filters: Dictionary of filters (e.g. {'gender': ['Men'], 'baseColour': ['Red']})
        
    Returns:
        pd.DataFrame: Top-k matching products dataframe with a 'score' column.
    """
    # Reshape embedding to (1, 512)
    q_emb = query_embedding.reshape(1, -1).astype('float32')
    
    # Check if we have active filters
    active_filters = {k: v for k, v in filters.items() if v} if filters else {}
    
    if not active_filters:
        # Fast path: Search the entire catalog using the FAISS index
        scores, idxs = index.search(q_emb, min(top_k, len(sample_df)))
        
        # Guard against indices that might be out of bounds
        valid_indices = [idx for idx in idxs[0] if 0 <= idx < len(sample_df)]
        valid_scores = scores[0][:len(valid_indices)]
        
        results = sample_df.iloc[valid_indices].copy()
        results["score"] = valid_scores
        return results
    
    else:
        # Filter path: filter dataframe first, then compute similarities on the remaining items
        filtered_mask = pd.Series(True, index=sample_df.index)
        
        for col, selected_vals in active_filters.items():
            if selected_vals:
                filtered_mask &= sample_df[col].isin(selected_vals)
                
        filtered_df = sample_df[filtered_mask].copy()
        
        if filtered_df.empty:
            return pd.DataFrame()
            
        # Perform matrix multiplication on the filtered subset of embeddings
        cand_indices = filtered_df.index.to_numpy()
        cand_embs = embeddings[cand_indices] # Shape: (M, 512)
        
        # Calculate dot products
        similarities = (cand_embs @ q_emb.T).flatten() # Shape: (M,)
        
        filtered_df["score"] = similarities
        # Sort and take top_k
        results = filtered_df.sort_values(by="score", ascending=False).head(top_k)
        return results

def get_complementary_recommendations(
    product_row: pd.Series,
    sample_df: pd.DataFrame,
    image_embeddings: np.ndarray,
    model: Any,
    tokenizer: Any,
    device: str,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Generates complementary product recommendations based on a query product
    and its category mappings, following the logic of the source notebook.
    
    Args:
        product_row: A pandas Series representing the input product.
        sample_df: The sample catalog.
        image_embeddings: Image embeddings matrix.
        model: CLIP model.
        tokenizer: CLIP tokenizer.
        device: Device string.
        top_k: Number of recommendations to return.
    """
    sub_cat = product_row.get("subCategory", "")
    master_cat = product_row.get("masterCategory", "")
    
    # 1. Main complementary mapping table
    complement_map = {
        "Shoes": ["Socks", "Watches", "Bags", "Sports Accessories", "Sunglasses"],
        "Topwear": ["Bottomwear", "Belts", "Watches", "Ties", "Necklace"],
        "Bottomwear": ["Topwear", "Belts", "Shoes"],
        "Dress": ["Shoes", "Bags", "Jewellery", "Earrings"],
        "Watches": ["Wallets", "Belts", "Sunglasses"],
        "Eyewear": ["Watches", "Wallets", "Belts", "Bags"],
        "Bags": ["Wallets", "Watches", "Belts"],
        "Innerwear": ["Loungewear and Nightwear", "Socks"],
        "Sandal": ["Socks", "Bags", "Sunglasses"],
        "Flip Flops": ["Bags", "Sunglasses"],
        "Belts": ["Topwear", "Bottomwear", "Watches"],
        "Socks": ["Shoes", "Sandal"],
        "Jewellery": ["Dress", "Topwear", "Bags"],
        "Wallets": ["Watches", "Belts", "Bags"],
        "Sports Accessories": ["Shoes", "Sports Equipment"],
        "Ties": ["Topwear", "Belts", "Watches"],
        "Loungewear and Nightwear": ["Innerwear", "Sandal"],
        "Saree": ["Jewellery", "Bags"],
        "Fragrance": ["Watches", "Wallets"],
        "Perfumes": ["Watches", "Wallets"],
        "Cufflinks": ["Topwear", "Ties", "Watches"],
        "Umbrellas": ["Bags", "Watches"],
    }

    # 2. Category-level fallback mapping
    masterCategory_fallback = {
        "Apparel": ["Accessories", "Footwear"],
        "Footwear": ["Accessories", "Apparel"],
        "Accessories": ["Apparel", "Footwear"],
        "Personal Care": ["Accessories"],
    }
    
    target_categories = complement_map.get(sub_cat, [])
    candidates = sample_df[sample_df["subCategory"].isin(target_categories)]
    
    # Fallback 1: Use masterCategory pairing
    if candidates.empty:
        fallback_masters = masterCategory_fallback.get(master_cat, [])
        candidates = sample_df[
            sample_df["masterCategory"].isin(fallback_masters) & (sample_df["subCategory"] != sub_cat)
        ]
        
    # Fallback 2: Just exclude the same subCategory, search everything else
    if candidates.empty:
        candidates = sample_df[sample_df["subCategory"] != sub_cat]
        
    if candidates.empty:
        return pd.DataFrame()
        
    # 3. Create descriptive search query text for CLIP text encoder
    query_text = f"{product_row.get('baseColour', '')} {product_row.get('articleType', '')} {product_row.get('usage', '')}".strip()
    
    tokens = tokenizer([query_text]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.cpu().numpy()
    
    # Calculate similarity with candidate images
    cand_idx = candidates.index.to_numpy()
    cand_embs = image_embeddings[cand_idx]
    similarities = (cand_embs @ text_emb.T).flatten()
    
    candidates = candidates.copy()
    candidates["score"] = similarities
    return candidates.sort_values(by="score", ascending=False).head(top_k)
