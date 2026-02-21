# GenAI Mid-Sem: NLP Research Analysis System

This is a traditional NLP-based system (Milestone 1) that analyzes documents (PDF, DOCX, TXT) and searches topics on Wikipedia to extract themes, keywords, and structured summaries without using Large Language Models (LLMs).

## Features
- **Extractive Summary**: Summarizes text using the TextRank algorithm.
- **Keywords (TF-IDF)**: Extracts the top relevant keywords and Visualizes them.
- **Topic Modeling (LDA)**: Uses Latent Dirichlet Allocation to identify underlying topics.
- **Document Support**: Upload PDF, DOCX, or TXT documents for analysis.
- **Wikipedia Search**: Automatically fetch and analyze topics directly from Wikipedia.

## Getting Started

1. **Install Requirements**:
   It is recommended to use a virtual environment. Install dependencies via:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the App**:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to the address shown in your terminal (usually `http://localhost:8501`).

## Project Structure
- `app.py`: Main Streamlit application file.
- `src/`: Contains source code for logic.
  - `data_fetcher.py`: Document text extraction and Wikipedia fetching.
  - `nlp_pipeline.py`: NLP processing functions (cleaning, TF-IDF, LDA, TextRank).
- `requirements.txt`: Python package dependencies.
