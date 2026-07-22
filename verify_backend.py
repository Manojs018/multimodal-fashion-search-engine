import os
import pandas as pd
import numpy as np
from PIL import Image
import utils

def test_system():
    workspace_dir = r"c:\Users\Manoj\OneDrive\Desktop\Gen_AI"
    models_dir = os.path.join(workspace_dir, "models")
    index_path = os.path.join(models_dir, "fashion_index.faiss")
    emb_path = os.path.join(models_dir, "image_embeddings.npy")
    df_path = os.path.join(models_dir, "sample_df.pkl")
    
    print("=== Testing Asset Availability ===")
    print("FAISS Index exists:", os.path.exists(index_path))
    print("Embeddings exist:", os.path.exists(emb_path))
    print("Metadata DataFrame exists:", os.path.exists(df_path))
    
    if not (os.path.exists(index_path) and os.path.exists(emb_path) and os.path.exists(df_path)):
        print("\n❌ Error: Missing assets! Please run setup_data.py first.")
        return
        
    print("\n=== Loading CLIP model and search assets ===")
    model, preprocess, tokenizer, device = utils.load_clip_model()
    index = utils.load_faiss_index(index_path)
    embeddings = utils.load_embeddings(emb_path)
    sample_df = utils.load_dataframe(df_path)
    
    print("CLIP Model Loaded successfully.")
    print(f"FAISS Index Loaded successfully with {index.ntotal} items.")
    print(f"Image Embeddings Loaded successfully with shape {embeddings.shape}.")
    print(f"Metadata DataFrame Loaded successfully with {len(sample_df)} items.")
    
    print("\n=== Testing Text Search ===")
    query_text = "red sneakers"
    print(f"Query: '{query_text}'")
    text_emb = utils.get_text_embedding(query_text, model, tokenizer, device)
    results = utils.search_products(text_emb, index, sample_df, embeddings, top_k=5)
    
    if results.empty:
        print("❌ No results found!")
    else:
        print("Top results:")
        for idx, row in results.iterrows():
            print(f"- ID: {row['id']} | {row['productDisplayName']} | Category: {row['articleType']} | Score: {row['score']:.4f}")
            
    print("\n=== Testing Image Search ===")
    # Pick the first image from sample_df as a test image
    test_row = sample_df.iloc[0]
    test_img_path = os.path.join(workspace_dir, test_row["image_path"])
    print(f"Query Image Path: '{test_img_path}' ({test_row['productDisplayName']})")
    
    if not os.path.exists(test_img_path):
        print(f"❌ Test image not found at {test_img_path}")
        return
        
    img = Image.open(test_img_path)
    img_emb = utils.get_image_embedding(img, model, preprocess, device)
    img_results = utils.search_products(img_emb, index, sample_df, embeddings, top_k=5)
    
    if img_results.empty:
        print("❌ No results found for image search!")
    else:
        print("Top visually similar results:")
        for idx, row in img_results.iterrows():
            print(f"- ID: {row['id']} | {row['productDisplayName']} | Category: {row['articleType']} | Score: {row['score']:.4f}")
            
    print("\n=== Testing Filters ===")
    filters = {"gender": ["Men"], "baseColour": ["Black"]}
    print(f"Applying filters: {filters}")
    filtered_results = utils.search_products(text_emb, index, sample_df, embeddings, top_k=5, filters=filters)
    
    if filtered_results.empty:
        print("No results (or none matched filters)")
    else:
        for idx, row in filtered_results.iterrows():
            print(f"- ID: {row['id']} | {row['productDisplayName']} | Gender: {row['gender']} | Colour: {row['baseColour']} | Score: {row['score']:.4f}")
            
    print("\n[SUCCESS] Verification Completed successfully!")

if __name__ == "__main__":
    test_system()
