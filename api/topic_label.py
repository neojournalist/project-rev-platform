"""
topic_label.py.py

Helper functions for topic labelling and filtering.

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

def get_domain():
  domains = [
        "Social Sciences",
        "Physical Sciences",
        "Health Sciences",
        "Life Sciences"
        ]

  print("Available categories:")
  for i, dom in enumerate(domains, 1):
    print(f"{i}. {dom}")

  domain = input("\nCopy and paste a category in here: ")
  return domain.strip()

def process_manuscript_and_filter_reviewers(manuscript_abstract, classifier, mapping_csv_path="topic_mapping.csv", reviewers_csv_path="reviewers_with_topic_labels.csv"):
    """
    Process a submitted manuscript abstract and filter reviewers by domain.
    
    Args:
        manuscript_abstract (str): The abstract text to be labeled
        classifier: The classifier object for labeling abstracts
        mapping_csv_path (str): Path to the topic mapping CSV file
        reviewers_csv_path (str): Path to the reviewers CSV file
    
    Returns:
        tuple: (mapping_dict, reviewers_df) where mapping_dict contains topic mapping info
               and reviewers_df is the filtered reviewers dataframe
    """
    # 2) Label submitted abstract and derive fields
    mapping_df = pd.read_csv(mapping_csv_path)  # must contain a 'topic_id' column!
    label = label_submitted_abstract(manuscript_abstract, classifier)
    mapping = map_topic_by_id(mapping_df, label)
    if mapping:
        print(f"→ Found mapping for ID {mapping['topic_id']} ('{mapping['matched_topic_name']}'):")
        print(f"   • Domain:   {mapping['domain_name']}")
        print(f"   • Keywords: {', '.join(mapping['keywords'][:3])}…")
        print(f"   • Summary:  {mapping['summary'][:100]}…")
    else:
        print("No mapping; please check that topic_id exists in your CSV.")

    # 3) Input a category or topic field
    domain = get_domain()

    # 4) Load the reviewer dataset and filter for the domain
    full_df = load_reviewers(reviewers_csv_path)
    reviewers_df = full_df[full_df["primary_topic.domain.display_name"] == domain]
    
    return mapping, reviewers_df