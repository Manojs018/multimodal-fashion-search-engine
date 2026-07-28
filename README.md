# Fashion Search & Recommendation System 🛍️

A production-ready, high-performance web application that leverages Deep Learning and Vector Databases to perform semantic text searches and visual similarity searches on fashion products. Built using **OpenCLIP (ViT-B-32)**, **FAISS**, **PyTorch**, and **Streamlit**.

---

## 🚀 Project Overview

This project converts a research-focused Jupyter notebook into a modular, production-ready, containerized web application. It uses:
- **OpenCLIP (ViT-B-32)**: For embedding both text and images into a shared multi-modal vector space.
- **FAISS (Facebook AI Similarity Search)**: For sub-second vector search over pre-computed image embeddings.
- **Streamlit**: For a gorgeous, interactive, glassmorphic UI optimized for both desktop and mobile.

---

## ✨ Features

- **💬 Semantic Text Search**: Search products naturally (e.g., `"red sneakers"` or `"blue casual shirt"`) and get relevant visual matches.
- **🖼️ Visual Similarity Search**: Upload any image to find visually matching products instantly from the catalog.
- **⚙️ Dynamic Metadata Filters**: Narrow down search results by Gender, Category, Sub-Category, Base Colour, and Season.
- **🌟 Outfit Recommender**: Click the **✨ Outfits** button on any product card to get complementary style recommendations (e.g., matching topwear with bottomwear/shoes) based on visual similarities.
- Performance Optimized: Fully cached models and database lookups ensure all search queries run in **< 1 second**.
- **📱 Premium Responsive Design**: A high-fidelity dark-themed UI built using custom CSS, grid card layouts, and progress bars.

---

## 🛠️ Tech Stack & Architecture

- **Frontend**: Streamlit, HTML5, Custom CSS3
- **Vector Embeddings**: PyTorch, OpenCLIP (ViT-B-32 pretrained on OpenAI)
- **Vector Search Engine**: FAISS-CPU
- **Data Manipulation**: Pandas, NumPy, Pillow (PIL)
- **Clustering (Notebook-only)**: Scikit-learn (Agglomerative Clustering)
- **Deployment**: Docker, Streamlit Cloud, Hugging Face, Render, Railway

```
                                  [ User Query ]
                                  /            \
                       (Text Input)            (Image Upload)
                            |                        |
                   [ CLIP Text Encoder ]     [ CLIP Image Encoder ]
                            \                        /
                             \                      /
                            [ Normalized Query Vector ]
                                         |
                                         v
                     [ FAISS / Cosine Similarity Filtered Search ]
                                         |
                                         v
                        [ Top 10 Product Matches & Scores ]
```

---

## 📂 Project Structure

```
project/
│
├── app.py                  # Streamlit Web Application (Frontend + View Logic)
├── utils.py                # Reusable backend functions (Model Loaders, Search, Recs)
├── setup_data.py           # Setup script to download dataset & pre-compute embeddings
├── requirements.txt        # Python package dependencies
├── Dockerfile              # Docker container setup configuration
├── README.md               # Project documentation
├── .gitignore              # Files ignored in git tracking
│
├── models/                 # Serialized model assets
│   ├── fashion_index.faiss # Pre-computed FAISS vector index
│   ├── image_embeddings.npy# Raw numpy matrix of image embeddings
│   └── sample_df.pkl       # Pickled pandas dataframe of product metadata
│
└── images/                 # Sampled product images (3,000 files)
```

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fashion-image-search.git
cd fashion-image-search
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets and build index
Run the setup script. This script automatically downloads the dataset from Kaggle, samples 3,000 products, copies their images into the project, and builds the FAISS vector index:
```bash
python setup_data.py
```

---

## 🏃 Running the Application

### Running Locally
Start the Streamlit application using the local environment:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### Running with Docker
You can containerize the application and run it without local python setup.

1. **Build the Docker Image**:
   ```bash
   docker build -t fashion-search-app .
   ```
2. **Run the Container**:
   ```bash
   docker run -p 8501:8501 fashion-search-app
   ```
Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 Deployment Instructions

### 1. Streamlit Community Cloud (Recommended)
1. Commit and push the project files to a public GitHub repository. (Note: Include `models/` and `images/` directory in your commit so the application has the pre-computed assets immediately).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in.
3. Click **New app**, select your repository, branch, and set the Main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will build the requirements and deploy the app.

### 2. Hugging Face Spaces (Docker Engine)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Docker** as the SDK.
3. Hugging Face Spaces will automatically detect the `Dockerfile` in your repository.
4. Git push your project files (including the pre-computed `models/` and `images/` folder) to the Hugging Face repository space.
5. Hugging Face will build the container and launch your application.

### 3. Render
1. Connect your GitHub repository to [Render](https://render.com/).
2. Create a **Web Service**.
3. Choose **Docker** as the runtime environment.
4. Render will use the repository's `Dockerfile` to build and serve the app. Set the start command to default and bind the port (Render handles port bindings automatically).

### 4. Railway
1. Sign in to [Railway.app](https://railway.app/).
2. Click **New Project** and select **Deploy from GitHub repo**.
3. Railway will read the `Dockerfile` and deploy the application.
4. Under **Variables**, define a port variable if needed (Railway assigns ports dynamically).

---

##  Future Improvements

1. **Incremental Indexing**: Add a backend admin panel to allow adding new products and computing their embeddings on the fly without rebuilds.
2. **Hybrid Search**: Combine lexical search (TF-IDF/BM25) with semantic vector search (CLIP) to handle specific SKU or barcode lookups.
3. **User Authentication**: Implement user logins and personalized boards to save favorite outfits.
4. **Cloud Database Storage**: Move from local FAISS/Pickle files to a cloud vector database like Milvus, Pinecone, or Qdrant for scaling to millions of products.
