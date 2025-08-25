import time
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import requests


def fetch_author_summary(author_name, email=None, top_n_topics=5):
    if not author_name or not author_name.strip():
        return {}

    headers = {"User-Agent": f"author-summary/1.0 ({email or 'no-email-provided'})"}
    params = {"search": author_name, "per_page": 1, "sort": "cited_by_count:desc"}
    if email:
        params["mailto"] = email

    try:
        r = requests.get("https://api.openalex.org/authors", params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json() or {}
        results = data.get("results", []) or []
        if not results:
            return {}

        aid = (results[0].get("id") or "").split("/")[-1]
        if not aid:
            return {}

        params2 = {}
        if email:
            params2["mailto"] = email

        r2 = requests.get(f"https://api.openalex.org/authors/{aid}", params=params2, headers=headers, timeout=10)
        r2.raise_for_status()
        author = r2.json() or {}

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


def _norm_inst(s):
    return (s or "").strip().lower()


def compute_coworker_conflicts(
    top_authors,
    submitted_author_institutions,
    email=None,
    rate_limit=0.3,
    top_n_topics=5
):
    profiles = []
    for rec in top_authors:
        if not rec:
            continue
        key = (rec or "").strip()
        if not key:
            continue

        profile = fetch_author_summary(author_name=key, email=email, top_n_topics=top_n_topics)

        if profile:
            profiles.append(profile)

        if rate_limit and rate_limit > 0:
            time.sleep(rate_limit)

    author_inst_df = pd.DataFrame(profiles)

    submit_insts = {_norm_inst(inst) for inst in submitted_author_institutions if inst}
    if "institution" not in author_inst_df:
        author_inst_df["institution"] = ""

    mask = author_inst_df["institution"].fillna("").map(_norm_inst).isin(submit_insts)
    coworker_conflicts = (
        author_inst_df.loc[mask, ["author", "institution"]]
        .to_dict(orient="records")
    )

    return author_inst_df, coworker_conflicts


def find_author_conflicts(df_remaining, submitted_authors):
    if not submitted_authors:
        return []

    authors_series = (
        df_remaining['authorships.author.display_name']
        .fillna('')
        .astype(str)
        .apply(lambda s: [name.strip() for name in s.split('|') if name.strip()])
    )

    mlb = MultiLabelBinarizer(sparse_output=True)
    author_matrix = mlb.fit_transform(authors_series)
    author_names = mlb.classes_

    name_to_idx = {name: idx for idx, name in enumerate(author_names)}
    submitted_idxs = [name_to_idx[name] for name in submitted_authors if name in name_to_idx]

    if not submitted_idxs:
        print("Submitted authors are not in the database. Detecting COI is impossible.")
        return []

    if not hasattr(author_matrix, 'tocsr'):
        author_matrix = author_matrix.tocsr()

    coauthor_counts = author_matrix.T.dot(author_matrix)
    coauthor_counts.setdiag(0)
    coauthor_counts.eliminate_zeros()

    direct_conflicts = (coauthor_counts[submitted_idxs, :].sum(axis=0) > 0).nonzero()[1]

    submitted_idxs_set = set(submitted_idxs)
    direct_idxs = set(direct_conflicts) - submitted_idxs_set

    return [author_names[i] for i in direct_idxs]


def filter_authors(top_authors, direct_conflicts, coworker_conflicts, submitted_authors):
    clean_authors = []
    rejected_authors = []

    coworker_author_names = set()
    try:
        coworker_author_names = {c.get('author') for c in coworker_conflicts if isinstance(c, dict)}
    except Exception:
        coworker_author_names = set()

    for entry in top_authors:
        author = entry['author']

        if author in submitted_authors:
            conflict_type = 'submitted_author'
        elif author in direct_conflicts:
            conflict_type = 'direct'
        elif author in coworker_author_names:
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


