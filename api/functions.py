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
import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
# Coworker filter with API openalex request
import requests
import time
from nltk.stem import PorterStemmer


def load_reviewers(path='clean_data_aug.csv'):
    """
    Load reviewer metadata from a CSV.
    """
    return pd.read_csv(path)

def load_topic_classifier(
    model_name = "OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract"
):
    """
    Load the OpenAlex topic classification pipeline.
    Returns a HuggingFace text-classification pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return classifier


def label_submitted_abstract(
    abstract,
    classifier
):
    """
    Generate a topic label for a single submitted abstract.
    Uses the same OpenAlex classifier pipeline.
    """
    result = classifier(abstract, truncation=True)
    if isinstance(result, list) and 'label' in result[0]:
        return result[0]['label']
    elif isinstance(result, dict) and 'label' in result:
        return result['label']
    else:
        raise ValueError(f"Unexpected classifier output: {result}")


def label_all_abstracts(
    df,
    classifier,
    text_col='abstract',
    batch_size = 16
):
    """
    Generate topic labels for all abstracts in a DataFrame.
    Adds or overwrites a 'topic_label' column based on the classifier.
    """
    texts = df[text_col].fillna('').tolist()
    labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results = classifier(
            batch,
            truncation=True
        )
        for res in results:
            if isinstance(res, list) and 'label' in res[0]:
                labels.append(res[0]['label'])
            elif isinstance(res, dict) and 'label' in res:
                labels.append(res['label'])
            else:
                labels.append('')
    new_df = df.copy()
    new_df['topic_label'] = labels
    return new_df


def load_model():
    # Load SPECTER2 model with adapter
    model = SentenceTransformer(
        "allenai/specter2_base",
        model_kwargs={'adapter_name': 'allenai/specter2'}
    )
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

def precompute_reviewer_embeddings(reviewers_df, emb_path='specter_embeddings.npy', model=None):
    """
    Compute or load reviewer embeddings.
    """
    if model is None:
        model = load_model()

    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)
    else:
        # Get valid abstracts
        valid_mask = reviewers_df['abstract'].notna()
        valid_df = reviewers_df[valid_mask]
        abstracts = valid_df['abstract'].tolist()
        
        # Format as empty title + abstract
        sep_token = model.tokenizer.sep_token
        text_batch = [f"{sep_token}{abstract}" for abstract in abstracts]
        
        # Generate embeddings
        embeddings = model.encode(
            text_batch,
            batch_size=32,
            show_progress_bar=True,
            device=model.device
        )
        np.save(emb_path, embeddings)
    
    return embeddings

def extract_topic_id_and_name(topic_label):
    """
    Extract numeric topic_id and clean topic_name from a manuscript label.
    E.g. "1147: The Spread of Misinformation Online" -> (1147, "The Spread of Misinformation Online")
    """
    m = re.match(r'^\s*(\d+)\s*:\s*(.+)$', topic_label)
    if m:
        topic_id = int(m.group(1))
        clean_name = m.group(2).strip()
    else:
        # fallback if no leading number found
        topic_id = None
        clean_name = topic_label.strip()
    print(f"DEBUG: Parsed '{topic_label}' -> topic_id={topic_id}, name='{clean_name}'")
    return topic_id, clean_name

def map_topic_by_id(mapping_df, topic_label):
    """
    Map manuscript topic label to domain hierarchy by topic_id.
    Falls back to warning if no ID match is found.
    """
    topic_id, clean_name = extract_topic_id_and_name(topic_label)
    if topic_id is None:
        print(f"WARNING: No numeric ID found in label '{topic_label}'.")
        return None

    # look up by topic_id
    match = mapping_df[mapping_df['topic_id'] == topic_id]

    if not match.empty:
        row = match.iloc[0]
        return {
            'topic_id':           row['topic_id'],
            'matched_topic_name': row['topic_name'],
            'subfield_name':      row['subfield_name'],
            'field_name':         row['field_name'],
            'domain_name':        row['domain_name'],
            'keywords':           row['keywords'].split(';') if isinstance(row['keywords'], str) else [],
            'summary':            row['summary']
        }
    else:
        print(f"WARNING: No mapping found for topic_id={topic_id} (label: '{clean_name}')")
        return None

def expertise_check_fast(manuscript_abstract, reviewers_df, reviewer_embeddings, top_n=10):
    """
    Fast reviewer matching using precomputed embeddings and clusters.
    Returns DataFrame with columns:
      ['id','similarity_score','euclidean_distance','abstract']
    """
    model = load_model()
    sep_token = model.tokenizer.sep_token
    
    # Format manuscript as empty title + abstract
    text = f"{sep_token}{manuscript_abstract}"

    # Generate manuscript embedding
    mv = model.encode([text], device=model.device)
    
    # Calculate similarities and distances
    sims = cosine_similarity(mv, reviewer_embeddings).flatten()
    dists = np.linalg.norm(reviewer_embeddings - mv, axis=1)

    # Combine results and sort by similarity score
    records = list(zip(range(len(sims)), sims, dists))
    records.sort(key=lambda x: x[1], reverse=True)

    # Get top N results
    top = records[:top_n]
    sel_idxs, sel_sims, sel_dists = zip(*top) if top else ([], [], [])

    # Prepare results DataFrame
    result = reviewers_df.iloc[list(sel_idxs)].copy()
    result['similarity_score'] = sel_sims
    result['euclidean_distance'] = sel_dists

    return result[['id','similarity_score','euclidean_distance','abstract', 
                   'primary_topic.domain.display_name', 'keywords.display_name', 
                   'topics.display_name']]

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
    """
    Search by author_name, then fetch the full author record by id.
    Returns a dict with keys: author, openalex_id, orcid, institution, subfields,
    works, cited, h_index, works_api_url, updated_date.
    """
    if not author_name or not author_name.strip():
        return {}

    headers = {"User-Agent": f"author-summary/1.0 ({email or 'no-email-provided'})"}
    params = {"search": author_name, "per_page": 1, "sort": "cited_by_count:desc"}
    if email:
        params["mailto"] = email

    try:
        # 1) search by name
        r = requests.get("https://api.openalex.org/authors", params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json() or {}
        results = data.get("results", []) or []
        if not results:
            return {}

        # 2) fetch the full author record by id (ensures institutions are present)
        aid = (results[0].get("id") or "").split("/")[-1]
        if not aid:
            return {}

        params2 = {}
        if email:
            params2["mailto"] = email

        r2 = requests.get(f"https://api.openalex.org/authors/{aid}", params=params2, headers=headers, timeout=10)
        r2.raise_for_status()
        author = r2.json() or {}

        # ---- institution resolution ----
        def resolve_institution(a):
            lki = a.get("last_known_institutions") or a.get("last_known_institution")
            if isinstance(lki, list) and lki:
                name = (lki[0].get("display_name") or "").strip()
                if name:
                    return name
            elif isinstance(lki, dict):
                name = (lki.get("display_name") or "").strip()
                if name:
                    return name

            best_name, best_year = "", -1
            for aff in a.get("affiliations", []) or []:
                inst = aff.get("institution") or {}
                name = (inst.get("display_name") or "").strip()
                years = aff.get("years") or []
                year = max(years) if years else -1
                if name and year > best_year:
                    best_name, best_year = name, year
            return best_name

        inst = resolve_institution(author)

        # ---- topics → subfields ----
        topics = sorted(author.get("topics", []), key=lambda t: t.get("count", 0), reverse=True)[:max(0, int(top_n_topics))]
        subfields = {
            (t.get("subfield") or {}).get("display_name", "").strip()
            for t in topics if t.get("subfield")
        }
        subfields_str = "; ".join(sorted(s for s in subfields if s))

        summary = author.get("summary_stats") or {}

        return {
            "author": author_name,
            "openalex_id": author.get("id", ""),
            "orcid": author.get("orcid"),
            "institution": inst,
            "subfields": subfields_str,
            "works": author.get("works_count", 0),
            "cited": author.get("cited_by_count", 0),
            "h_index": summary.get("h_index", 0),
            "works_api_url": author.get("works_api_url", ""),
            "updated_date": author.get("updated_date", ""),
        }

    except requests.RequestException as e:
        print(f"OpenAlex error for '{author_name}': {e}")
        return {}

# helper: normalize institution strings for matching
def _norm_inst(s):
    return (s or "").strip().lower()

# Coworker filter with fetch API to OpenAlex
def compute_coworker_conflicts(
    top_authors,
    submitted_author_institutions,
    email=None,
    rate_limit=0.3,
    top_n_topics=5
):
    """
    1) Call fetch_author_summary for each author (by name or OpenAlex ID/URL)
    2) Build a DF for all returned profiles
    3) Mark reviewers whose institution matches any submitted author's institution
    Returns: (author_institution_df, coworker_conflicts)
    """
    profiles = []
    for rec in top_authors:
        if not rec:
            continue
        key = (rec or "").strip()
        if not key:
            continue

        # detect if an OpenAlex author ID/URL was supplied
        profile = fetch_author_summary(author_name=key, email=email, top_n_topics=top_n_topics)

        if profile:
            profiles.append(profile)

        if rate_limit and rate_limit > 0:
            time.sleep(rate_limit)

    author_inst_df = pd.DataFrame(profiles)

    # ---------- identify coworker conflicts ----------
    submit_insts = {_norm_inst(inst) for inst in submitted_author_institutions if inst}
    # make sure the column exists even if no profiles
    if "institution" not in author_inst_df:
        author_inst_df["institution"] = ""

    mask = author_inst_df["institution"].fillna("").map(_norm_inst).isin(submit_insts)
    coworker_conflicts = (
        author_inst_df.loc[mask, ["author", "institution"]]
        .to_dict(orient="records")
    )

    return author_inst_df, coworker_conflicts


# Direct conflict Filtering
def find_author_conflicts(df_remaining, submitted_authors):
    """
    Args:
        df_remaining: DataFrame with 'authorships.author.display_name' column
        submitted_authors: List of author names to check conflicts for

    Returns:
        tuple: (direct_conflicts)
            - direct_conflicts: List of authors who co-authored with submitted authors
    """

    if not submitted_authors:
      return []

    # 1) Create author lists from pipe-separated strings
    df_remaining['authors'] = (df_remaining['authorships.author.display_name']
    .fillna('')
    .astype(str)
    .apply(lambda s: [name.strip() for name in s.split('|') if name.strip()]))

    #2) Create sparse author incidence matrix and mapping
    mlb = MultiLabelBinarizer(sparse_output=True)
    author_matrix = mlb.fit_transform(df_remaining['authors'])
    author_names = mlb.classes_

    # 3) Get valid submitted author indices
    name_to_idx = {name: idx for idx, name in enumerate(author_names)}
    submitted_idxs = [name_to_idx[name] for name in submitted_authors if name in name_to_idx]

    if not submitted_idxs:
      print("Submitted authors are not in the database. Detecting COI is impossible.")
      return []

    # 4) Compute co-authorship matrix efficiently
    # Ensure we're working with CSR format for optimal performance
    if not hasattr(author_matrix, 'tocsr'):
        author_matrix = author_matrix.tocsr()
    
    coauthor_counts = author_matrix.T.dot(author_matrix)
    coauthor_counts.setdiag(0)
    coauthor_counts.eliminate_zeros()

    # 5) Find direct conflicts
    direct_conflicts = (coauthor_counts[submitted_idxs, :].sum(axis=0) > 0).nonzero()[1]
   
    # 6) Exclude submitted authors from conflicts
    submitted_idxs_set = set(submitted_idxs)
    direct_idxs = set(direct_conflicts) - submitted_idxs_set

    return [author_names[i] for i in direct_idxs]


def filter_authors(top_authors, direct_conflicts, coworker_conflicts, submitted_authors):
    """
    Removes authors in conflict sets and returns clean + rejected lists.

    Parameters:
        top_authors: list of dicts with 'author', 'paper_id', etc.
        direct_conflicts: set of names with direct conflict
        submitted_authors: list or set of original submitted authors

    Returns:
        (clean_authors, rejected_authors)
    """
    clean_authors = []
    rejected_authors = []

    for entry in top_authors:
        author = entry['author']

        if author in submitted_authors:
            conflict_type = 'submitted_author'
        elif author in direct_conflicts:
            conflict_type = 'direct'
        elif author in coworker_conflicts:
            conflict_type = 'coworker'
        else:
            clean_authors.append(entry)
            continue

        rejected_authors.append({
            'author': author,
            'conflict_type': conflict_type,
            'paper_id': entry['paper_id']
        })

    return clean_authors, rejected_authors

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
    stemmer = PorterStemmer()
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