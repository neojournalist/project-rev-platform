#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions.py

Helper functions for reviewer recommendation.

Required input file:
  - clean_data.csv : CSV containing reviewer metadata. Must include columns:
      'id', 'abstract', 'authorships.author.display_name'

Also requires precomputed files created via these functions:
  - specter_embeddings.npy   : reviewer embeddings array

Dependencies:
  - numpy
  - pandas
  - torch
  - sentence_transformers
  - scikit-learn
  - requests
"""

import os
import numpy as np
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
# Coworker filter with API openalex request
import requests
import time
try:
    from nltk.stem import PorterStemmer  # optional dependency
except Exception:  # pragma: no cover - fallback if nltk isn't installed
    PorterStemmer = None


def load_reviewers(path='clean_data_aug.csv'):
    """
    Load reviewer metadata from a CSV.
    """
    return pd.read_csv(path)

def load_topic_classifier(model_name = "OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract"):
    # Backward-compat shim
    from .labeling import load_topic_classifier as _load
    return _load(model_name)


def label_submitted_abstract(abstract, classifier):
    from .labeling import label_submitted_abstract as _label_one
    return _label_one(abstract, classifier)


def label_all_abstracts(df, classifier, text_col='abstract', batch_size = 16):
    from .labeling import label_all_abstracts as _label_all
    return _label_all(df, classifier, text_col=text_col, batch_size=batch_size)


def load_model():
    # Backward-compat shim to embeddings module
    from .embeddings import load_model as _load
    return _load()

def precompute_reviewer_embeddings(reviewers_df, emb_path='specter_embeddings.npy', model=None):
    from .embeddings import precompute_reviewer_embeddings as _pre
    return _pre(reviewers_df, emb_path=emb_path, model=model)

def extract_topic_id_and_name(topic_label):
    from .labeling import extract_topic_id_and_name as _extract
    topic_id, clean_name = _extract(topic_label)
    print(f"DEBUG: Parsed '{topic_label}' -> topic_id={topic_id}, name='{clean_name}'")
    return topic_id, clean_name

def map_topic_by_id(mapping_df, topic_label):
    from .mapping import map_topic_by_id as _map
    return _map(mapping_df, topic_label)

def expertise_check_fast(manuscript_abstract, reviewers_df, reviewer_embeddings, top_n=10):
    from .similarity import expertise_check_fast as _fast
    return _fast(manuscript_abstract, reviewers_df, reviewer_embeddings, top_n=top_n)

def extract_top_authors(results_df, reviewers_df, author_col='authorships.author.display_name', top_n_authors=None):
    """
    Explode top matched papers into individual author records.
    """
    author_entries = []
    for _, row in results_df.iterrows():
        paper_id = row['id']
        abstract = row['abstract']
        domain = row['primary_topic.domain.display_name']
        similarity = row['similarity_score']
        distance = row['euclidean_distance']
        keywords = row['keywords.display_name']
        topics = row['topics.display_name']
        paper_row = reviewers_df[reviewers_df['id'] == paper_id]
        if paper_row.empty:
            continue
        authors_str = paper_row.iloc[0].get(author_col, '')
        if pd.isna(authors_str):
            continue
        authors = [a.strip() for a in authors_str.split('|') if a.strip()]
        for author in authors:
            author_entries.append({
                'author': author,
                'paper_id': paper_id,
                'abstract': abstract,
                'domain': domain,
                'keywords': keywords,
                'topics': topics,
                'similarity_score': similarity,
                'euclidean_distance': distance
            })
    if top_n_authors:
        seen = set()
        deduped = []
        for entry in sorted(author_entries, key=lambda x: -x['similarity_score']):
            if entry['author'] not in seen:
                deduped.append(entry)
                seen.add(entry['author'])
            if len(deduped) >= top_n_authors:
                break
        return deduped
    return author_entries

def fetch_author_summary(author_name, email=None, top_n_topics=5):
    # Backward-compat shim: import from conflicts module
    from .conflicts import fetch_author_summary as _fetch
    return _fetch(author_name=author_name, email=email, top_n_topics=top_n_topics)

# helper: normalize institution strings for matching
def _norm_inst(s):
    return (s or "").strip().lower()

# Coworker filter with fetch API to OpenAlex
def compute_coworker_conflicts(top_authors, submitted_author_institutions, email=None, rate_limit=0.3, top_n_topics=5):
    from .conflicts import compute_coworker_conflicts as _compute
    return _compute(top_authors=top_authors, submitted_author_institutions=submitted_author_institutions, email=email, rate_limit=rate_limit, top_n_topics=top_n_topics)


# Direct conflict Filtering
def find_author_conflicts(df_remaining, submitted_authors):
    from .conflicts import find_author_conflicts as _find
    return _find(df_remaining=df_remaining, submitted_authors=submitted_authors)


def filter_authors(top_authors, direct_conflicts, coworker_conflicts, submitted_authors):
    from .conflicts import filter_authors as _filter
    return _filter(top_authors=top_authors, direct_conflicts=direct_conflicts, coworker_conflicts=coworker_conflicts, submitted_authors=submitted_authors)

def normalize_log_percentile(s: pd.Series, p=75):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    ref = np.percentile(s, p) or 1.0
    return np.log1p(np.clip(s, 0, ref)) / np.log1p(ref)

def jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def to_token_set(x):
    """
    tokenization with:
    - Case normalization
    - Stemming
    - Better special character handling
    - More inclusive token length
    """
    stemmer = PorterStemmer() if PorterStemmer else None
    if x is None or (isinstance(x, float)) and pd.isna(x):
        return set()
    
    # Handle different input types
    if isinstance(x, str):
        items = [x]
    elif isinstance(x, (list, tuple, set)):
        items = x
    else:
        items = [str(x)]
    
    tokens = set()
    for item in items:
        if pd.isna(item):
            continue
            
        # Convert to lowercase and handle special cases
        text = str(item).lower()
        
        # Keep hyphens and apostrophes within words
        text = re.sub(r'(?<!\w)[\'\-](?!\w)', ' ', text)  # Handle standalone hyphens/apostrophes
        text = re.sub(r'[^\w\s\-\']', ' ', text)  # Keep hyphens and apostrophes within words
        
        # Tokenize while preserving intra-word hyphens/apostrophes
        words = re.findall(r'\b[\w\-\']+\b', text)
        
        # Stemming and filtering
        for word in words:
            # Keep all words except single characters (unless they're important)
            if len(word) == 1 and word not in {'c', 'r', 'r'}:
                continue
                
            stemmed = stemmer.stem(word) if stemmer else word
            tokens.add(stemmed)
            
    return tokens