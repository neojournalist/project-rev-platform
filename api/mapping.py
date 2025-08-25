import pandas as pd
from .labeling import extract_topic_id_and_name


def map_topic_by_id(mapping_df: pd.DataFrame, topic_label: str):
    topic_id, clean_name = extract_topic_id_and_name(topic_label)
    if topic_id is None:
        print(f"WARNING: No numeric ID found in label '{topic_label}'.")
        return None

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


