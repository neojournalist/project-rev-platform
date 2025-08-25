import os
import numpy as np
from .models import get_specter_model


def load_model():
    return get_specter_model()


def precompute_reviewer_embeddings(reviewers_df, emb_path='specter_embeddings.npy', model=None):
    if model is None:
        model = load_model()

    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)
    else:
        valid_mask = reviewers_df['abstract'].notna()
        valid_df = reviewers_df[valid_mask]
        abstracts = valid_df['abstract'].tolist()

        sep_token = model.tokenizer.sep_token
        text_batch = [f"{sep_token}{abstract}" for abstract in abstracts]

        embeddings = model.encode(
            text_batch,
            batch_size=32,
            show_progress_bar=True,
            device=model.device
        )
        np.save(emb_path, embeddings)
    return embeddings


