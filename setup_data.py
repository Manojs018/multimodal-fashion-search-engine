import os
import shutil
import pandas as pd
import numpy as np
import torch
import open_clip
import faiss
from PIL import Image
from tqdm import tqdm
import kagglehub

def main():
    # 1. Paths configuration
    workspace_dir = r"c:\Users\Manoj\OneDrive\Desktop\Gen_AI"
    models_dir = os.path.join(workspace_dir, "models")
    images_dest_dir = os.path.join(workspace_dir, "images")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dest_dir, exist_ok=True)
    
    print("Downloading fashion dataset using kagglehub...")
    # This downloads paramaggarwal/fashion-product-images-small (containing images/ and styles.csv)
    dataset_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
    print(f"Dataset downloaded to: {dataset_path}")
    
    csv_path = os.path.join(dataset_path, "styles.csv")
    img_src_dir = os.path.join(dataset_path, "images")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"styles.csv not found in downloaded dataset at {csv_path}")
    if not os.path.exists(img_src_dir):
        raise FileNotFoundError(f"images directory not found in downloaded dataset at {img_src_dir}")
        
    # Copy styles.csv to project root
    dest_csv_path = os.path.join(workspace_dir, "styles.csv")
    shutil.copy(csv_path, dest_csv_path)
    print(f"Copied styles.csv to project root: {dest_csv_path}")
    
    # 2. Read and preprocess CSV
    print("Reading styles.csv...")
    df = pd.read_csv(dest_csv_path, on_bad_lines='skip')
    print(f"Loaded {len(df)} rows from styles.csv.")
    
    # Add absolute source image paths to verify existence
    df["src_image_path"] = df["id"].astype(str).apply(lambda x: os.path.join(img_src_dir, f"{x}.jpg"))
    df = df[df["src_image_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Filtered rows with matching source images: {len(df)}")
    
    if len(df) == 0:
        raise ValueError("No images found matching product IDs in styles.csv!")
        
    # 3. Sample 3000 rows
    SAMPLE_SIZE = 3000
    sample_df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sample_df)} products for indexing.")
    
    # 4. Copy images to project images/ folder and set relative path
    print("Copying sampled images to project directory...")
    relative_paths = []
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        src_path = row["src_image_path"]
        prod_id = row["id"]
        dest_filename = f"{prod_id}.jpg"
        dest_path = os.path.join(images_dest_dir, dest_filename)
        
        # Copy file
        shutil.copy(src_path, dest_path)
        # Store relative path in project
        relative_paths.append(os.path.join("images", dest_filename))
        
    sample_df["image_path"] = relative_paths
    # Drop source image path column to keep it clean
    sample_df = sample_df.drop(columns=["src_image_path"])
    
    # 5. Initialize OpenCLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing OpenCLIP model (ViT-B-32) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device).eval()
    
    # 6. Extract embeddings
    print("Computing embeddings for sampled images...")
    embeddings = []
    batch_size = 64
    
    for i in tqdm(range(0, len(sample_df), batch_size)):
        batch_rows = sample_df.iloc[i:i+batch_size]
        imgs = []
        valid_indices = []
        
        for idx, row in batch_rows.iterrows():
            img_abs_path = os.path.join(workspace_dir, row["image_path"])
            try:
                img = Image.open(img_abs_path).convert("RGB")
                imgs.append(preprocess(img))
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading image {img_abs_path}: {e}")
                
        if not imgs:
            continue
            
        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            
        embeddings.append(feats.cpu().numpy())
        
    image_embeddings = np.vstack(embeddings)
    print(f"Embeddings generated with shape: {image_embeddings.shape}")
    
    # Save embeddings
    emb_path = os.path.join(models_dir, "image_embeddings.npy")
    np.save(emb_path, image_embeddings)
    print(f"Saved embeddings to {emb_path}")
    
    # 7. Create and build FAISS index
    print("Building FAISS index...")
    d = image_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(image_embeddings.astype('float32'))
    
    # Save FAISS index
    index_path = os.path.join(models_dir, "fashion_index.faiss")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path} with {index.ntotal} items.")
    
    # 8. Save sample DataFrame
    pkl_path = os.path.join(models_dir, "sample_df.pkl")
    sample_df.to_pickle(pkl_path)
    print(f"Saved sample DataFrame to {pkl_path}")
    
    print("Data setup completed successfully!")

if __name__ == "__main__":
    main()
