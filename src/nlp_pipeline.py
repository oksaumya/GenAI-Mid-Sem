import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

# Initialize spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text) # Remove extra whitespace
    text = re.sub(r'\[[0-9]*\]', ' ', text) # Remove wiki citations
    return text.strip()

def preprocess_text(text):
    """
    Tokenization, Stop-word removal, Lemmatization using spaCy.
    Returns a cleaned string ready for TF-IDF.
    """
    # Use spacy for processing, limited size to avoid memory issues for very large docs
    if len(text) > 1000000:
        text = text[:1000000]
    
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha:
            tokens.append(token.lemma_.lower())
    
    return " ".join(tokens)

def extract_keywords_tfidf(text, top_n=10):
    """Extract top keywords using TF-IDF."""
    preprocessed = preprocess_text(text)
    if not preprocessed.strip():
        return []
    
    # Needs a list of documents, we treat each sentence as a document
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    cleaned_sentences = [preprocess_text(s) for s in sentences if s.strip()]
    
    if not cleaned_sentences:
        return []

    vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.01)
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum tfidf scores across all sentences
        sum_scores = tfidf_matrix.sum(axis=0)
        scores = [(word, sum_scores[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    except ValueError:
        return [] # Can happen if vocabulary is empty

def perform_topic_modeling(text, n_topics=3, n_words=5):
    """Topic modeling using LDA."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    cleaned_sentences = [preprocess_text(s) for s in sentences if len(s.split()) > 3]
    
    if len(cleaned_sentences) < n_topics:
        return [] # Need more sentences for topic modeling

    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[: -n_words - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append(top_features)
        return topics
    except ValueError:
        return []

def generate_extractive_summary(text, sentences_count=5):
    """Generate summary using TextRank (from sumy)."""
    if len(text.split()) < 50:
        return text # Too short to summarize

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return "\n".join([str(sentence) for sentence in summary])
    except Exception as e:
        return f"Error generating summary: {e}"
