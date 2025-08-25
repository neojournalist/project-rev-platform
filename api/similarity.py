import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .embeddings import load_model


def expertise_check_fast(manuscript_abstract, reviewers_df, reviewer_embeddings, top_n=10):
    model = load_model()
    sep_token = model.tokenizer.sep_token

    text = f"{sep_token}{manuscript_abstract}"
    mv = model.encode([text], device=model.device)

    sims = cosine_similarity(mv, reviewer_embeddings).flatten()
    dists = np.linalg.norm(reviewer_embeddings - mv, axis=1)

    records = list(zip(range(len(sims)), sims, dists))
    records.sort(key=lambda x: x[1], reverse=True)

    top = records[:top_n]
    sel_idxs, sel_sims, sel_dists = zip(*top) if top else ([], [], [])

    result = reviewers_df.iloc[list(sel_idxs)].copy()
    result['similarity_score'] = sel_sims
    result['euclidean_distance'] = sel_dists

    return result[['id','similarity_score','euclidean_distance','abstract', 
                   'primary_topic.domain.display_name', 'keywords.display_name', 
                   'topics.display_name']]


