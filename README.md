# Project 1: Intelligent Research Topic Analysis & Agentic AI Research Assistant

## From Traditional NLP to Autonomous Agents

### Project Overview
This project involves the design and implementation of an AI-based research automation system.

#### **Milestone 1:** 
- Students build a traditional machine learning and NLP-based research analysis system that processes multiple documents related to a topic, performs text preprocessing, keyword extraction, topic modeling, and basic summarization.

**Features**
- **Extractive Summary**: Summarizes text using the TextRank algorithm.
- **Keywords (TF-IDF)**: Extracts the top relevant keywords and Visualizes them.
- **Topic Modeling (LDA)**: Uses Latent Dirichlet Allocation to identify underlying topics.
- **Document Support**: Upload PDF, DOCX, or TXT documents for analysis.
- **Wikipedia Search**: Automatically fetch and analyze topics directly from Wikipedia.

#### **Milestone 2:** 
- The same system is extended into a full agentic AI research assistant that accepts open-ended research queries, autonomously retrieves information from the web, reasons across multiple sources, manages state using LangGraph, and generates structured research reports.

The project emphasizes progression from classical NLP and ML concepts to agentic AI workflows, LLM orchestration, and deployment.

---

### Constraints & Requirements
- **Team Size:** 3 Students
- **API Budget:** Free Tier Only
- **Framework:** LangGraph (Recommended) Mandatory
- **Hosting:** Mandatory (Hugging Face Spaces, Streamlit Community Cloud, Render)

---

### Technology Stack
| Component | Technology |
| :--- | :--- |
| **LLMs (Free Only)** | Hugging Face (Open Source), Ollama (Local Models), Free-tier LLM APIs |
| **Web Search & Retrieval** | Tavily (Free Tier), DuckDuckGo Search Wrappers, Bing Web Search (Free Quota) |
| **NLP Libraries (M1)** | NLTK, spaCy, Sumy,Gensim (Topic Modeling), Scikit-learn (TF-IDF/ML) |
| **UI Framework** | Streamlit (Recommended), Gradio |
| **Hosting Platform** | Hugging Face Spaces, Streamlit Community Cloud, Render (Free Tier) |

---

### Milestones & Deliverables

#### Milestone 1: NLP Research Analysis System (Local) (Mid-Sem)
**Objective:** Design and implement a traditional NLP-based system that analyzes documents related to a research topic and produces structured analytical summaries without using LLMs or agentic workflows.

**Functional Requirements:**
- Accept a research topic or document collection.
- Perform text preprocessing and feature extraction.
- Identify key themes and keywords.
- Generate extractive or statistical summaries.
- Display analytical results through a basic UI.

**Technical Requirements:**
- **Inputs & Outputs:** Input: Research topic keywords or uploaded documents. Output: Key terms/themes, Topic clusters, Extractive summary. Optional: Analytical visualizations.
- **Preprocessing:** Tokenization, Stop-word removal, Lemmatization.
- **Feature Extraction:** TF-IDF or Bag-of-Words.
- **Modeling:** Topic modeling (LDA) or Clustering (K-Means).
- **Summarization:** Basic Extractive techniques.
- **Evaluation:** Coherence or interpretability measures.

**Key Deliverables:**
- Problem understanding & use-case description.
- Input–output specification.
- System architecture diagram (Traditional NLP).
- ML/NLP implementation with explanations.
- Working local application with Basic UI.
- Short report describing limitations of traditional approaches.

#### Milestone 2: Agentic AI Research Assistant (Web) (End-Sem)
**Objective:** Extend the research analysis system into a fully autonomous research agent that can retrieve information, reason across sources, manage state, and generate structured research reports.

**Functional Requirements:**
- Accept open-ended research queries.
- Perform web search and retrieval.
- Aggregate information from multiple sources.
- Maintain state across research steps.
- Generate structured research reports.

**Technical Requirements:**
- **LLM Integration:** Open-source/Free-tier only.
- **Workflow:** LangGraph nodes (Search, Summary, Validate). Edges: Logic for multi-step control flow. State: Explicit state management. Robustness: Handle API failures, noisy sources.
- **Structured Output:** Content: Title, Abstract, Key Findings, Conclusion. Sources: URLs and clear attribution. Quality: Evidence-based summaries to reduce hallucinations.

**Key Deliverables:**
- Publicly hosted application link.
- Agentic AI workflow using LangGraph.
- Structured research report generation.
- Enhanced UI with status indicators.
- Github Repository & Demo Video (5-7 mins).
- Extension (Pick 1): Follow-up Q&A, Topic Expansion, PDF Export, or Session Memory.

---

### Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem (Milestone 1)** | 25% | Correctness of NLP Pipeline (Preprocessing, LDA, TF-IDF), Quality of Topic Modeling & Keyword Extraction. |

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
