# api.py
from flask import jsonify
import re
import os
import pandas as pd
import numpy as np
from .functions import (
    load_reviewers,
    load_model,
    load_topic_classifier,
    expertise_check_fast,
    extract_top_authors,
    label_submitted_abstract,
    map_topic_by_id,
    find_author_conflicts,
    compute_coworker_conflicts,
    filter_authors,
    normalize_log_percentile,
    to_token_set,
    jaccard
)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def run_expertise_pipeline(manuscript_abstract, submitted_authors, submitted_institutions, domain=None):
    print("🚀 Starting expertise pipeline...")
    
    print("📊 Loading reviewers data...")
    base_dir = os.path.dirname(__file__)
    reviewers_path = os.path.join(base_dir, "clean_data_aug.csv")
    reviewers_df = load_reviewers(reviewers_path)
    print(f"   Loaded {len(reviewers_df)} reviewers")
    
    print("🧠 Loading embeddings...")
    emb_path = os.path.join(base_dir, "specter_embeddings.npy")
    embeddings = np.load(emb_path)
    print(f"   Loaded embeddings shape: {embeddings.shape}")
    
    print("🏷️ Loading topic classifier...")
    classifier = load_topic_classifier()
    
    print("📝 Labeling submitted abstract...")
    submitted_label = label_submitted_abstract(manuscript_abstract, classifier)
    print(f"   Topic label: {submitted_label}")
    
    # 2) Label sunmitted abstract and derive fields
    print("🗺️ Loading topic mapping...")
    mapping_csv = os.path.join(base_dir, "topic_mapping.csv")  # must contain a 'topic_id' column!
    mapping_df = pd.read_csv(mapping_csv)
    label = label_submitted_abstract(manuscript_abstract, classifier)
    mapping = map_topic_by_id(mapping_df, label)
    print(f"   Topic mapping: {mapping}")
    
    print("🔍 Finding similar reviewers...")
    results = expertise_check_fast(
        manuscript_abstract,
        reviewers_df,
        embeddings,
        top_n=100 # changed to 10 from 100
    )
    print(f"   Found {len(results)} similar reviewers")
    
    print("👥 Extracting top authors...")
    top_authors = extract_top_authors(
        results_df=results,
        reviewers_df=reviewers_df,
        author_col='authorships.author.display_name',
        top_n_authors=100
    )
    print(f"   Extracted {len(top_authors)} top authors")
    
    # Apply domain filter to top_authors if domain is specified
    if domain:
        print(f"🔍 Filtering by domain: {domain}")
        filtered_top_authors = []
        for rec in top_authors:
            # Try to get the domain from the record (may be under 'domain' or 'primary_topic.domain.display_name')
            rec_domain = rec.get('domain') or rec.get('primary_topic.domain.display_name')
            if rec_domain == domain:
                filtered_top_authors.append(rec)
        top_authors = filtered_top_authors
        print(f"   After domain filtering: {len(top_authors)} authors")
    
    author_names = [rec['author'] for rec in top_authors]
    
    print("⚠️ Checking for author conflicts...")
    direct_conflicts = find_author_conflicts(reviewers_df, submitted_authors)
    print(f"   Direct conflicts: {len(direct_conflicts)}")
    
    personal_email = os.getenv("email")

    print("🏢 Checking coworker conflicts...")
    author_inst_df, coworker_conflicts = compute_coworker_conflicts(
        top_authors=author_names,
        submitted_author_institutions=submitted_institutions,
        email=personal_email,
        rate_limit=0.3
    )
    print(f"   Coworker conflicts: {len(coworker_conflicts)}")
    
    print("🔍 Filtering authors...")
    filtered_authors, rejected_authors = filter_authors(
        top_authors,
        direct_conflicts,
        coworker_conflicts,
        submitted_authors
    )
    print(f"   Filtered authors: {len(filtered_authors)}, Rejected: {len(rejected_authors)}")
    
    print("📊 Processing final results...")
    authors_df = author_inst_df.copy()
    top_df = pd.DataFrame(top_authors) # Create a dataframe (not needed for the pipeline) 
    cols_merge = top_df[['paper_id', 'abstract', 'keywords', 'topics', 'author', 'similarity_score', 'euclidean_distance', 'domain']]
    final_df = pd.merge(authors_df, cols_merge, on='author', how='left')
    final_df["cited_norm"] = normalize_log_percentile(final_df["cited"], p=75)
    final_df["works_norm"] = normalize_log_percentile(final_df["works"], p=75)
    # Use keywords from mapping if available, otherwise use empty list
    A = mapping.get('keywords', []) if mapping else []
    # Convert A to a set for jaccard calculation
    A_set = to_token_set(A)
    final_df["kw_jaccard"] = final_df['keywords'].apply(lambda x: jaccard(A_set, to_token_set(x)))
    final_df['score'] = 0.5 * final_df['similarity_score'] + 0.2 * final_df['kw_jaccard'] + 0.2 * final_df['cited_norm'] + 0.1 * final_df['works_norm']
    final_df.sort_values(by='score', ascending=False, inplace=True)
    
    print("🧹 Cleaning results...")
    cleaned = []
    for _, rec in final_df.iterrows():
        author_name = str(rec.get("author")) if rec.get("author") is not None else None
        score = float(rec.get("score")) if rec.get("score") is not None else 0.0
        author_id = str(rec.get("openalex_id")) if rec.get("openalex_id") is not None else None
        paper_id = str(rec.get("paper_id")) if rec.get("paper_id") is not None else None
        similarity = float(rec.get("similarity_score")) if rec.get("similarity_score") is not None else 0.0
        cited = float(rec.get("cited_norm")) if rec.get("cited_norm") is not None else 0.0
        works = float(rec.get("works_norm")) if rec.get("works_norm") is not None else 0.0
        kw_jaccard = float(rec.get("kw_jaccard")) if rec.get("kw_jaccard") is not None else 0.0
        orcid = str(rec.get("orcid")) if rec.get("orcid") is not None else None
        abstract = str(rec.get("abstract")) if rec.get("abstract") is not None else None
        # Get domain from the record
        domain_info = rec.get("domain") or rec.get("primary_topic.domain.display_name") or "Unknown"
        cleaned.append({
            "author": author_name,
            "score": score,
            "author_id": author_id,
            "paper_id": paper_id,
            "similarity": similarity,
            "cited": cited,
            "works": works,
            "kw_jaccard": kw_jaccard,
            "orcid": orcid,
            "domain": domain_info,
            "abstract": abstract
            })
    
    print(f"✅ Pipeline completed! Returning {len(cleaned)} final authors")
    return convert_numpy_types({
        "label": submitted_label,
        "mapping": mapping,
        "authors": cleaned,
        "coi_results": {
            "direct_conflicts": direct_conflicts,
            "coworker_conflicts": coworker_conflicts,
            "rejected_authors": rejected_authors,
            "total_rejected": len(rejected_authors),
            "total_checked": len(top_authors)
        }
    })
