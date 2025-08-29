#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_expertise_check.py

This script ties together the reviewer recommendation pipeline.

Required files:
  - reviewers_with_topic_labels.csv               : CSV file with reviewer metadata, must include 'id' and 'abstract' columns.
  - submitted_manuscript.txt     : Plain text file containing the manuscript abstract.
  - submitted_authors.txt        : Plain text file listing submitted author names, one per line.
  - submitted_institutions.txt   : Plain text file listing submitted institution names, one per line.
  - specter_embeddings.npy      : Precomputed reviewer embeddings (from functions.precompute_reviewer_embeddings).

Output:
  - top_authors_with_institutions.csv : CSV with top author matches and metadata.

Dependencies:
  - pandas
  - functions.py module
"""


def main():
    # 1) Load model and classifier,
    classifier = load_topic_classifier()
    model = load_model()

    # 2) Label sunmitted abstract and derive fields
    mapping_df = pd.read_csv("topic_mapping.csv")  # must contain a 'topic_id' column!
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
    full_df = load_reviewers("reviewers_with_topic_labels.csv")
    reviewers_df = full_df[full_df["primary_topic.domain.display_name"] == domain]

    # 5) Load precomputed embeddings or compute
    reviewer_embeddings = precompute_reviewer_embeddings(reviewers_df, model=model)

    # 5) Expertise check
    results = expertise_check_fast(
        manuscript_abstract,
        reviewers_df,
        reviewer_embeddings
        )

    # 6) Extract top authors and conflicts
    top_authors = extract_top_authors(results, reviewers_df, author_col='authorships.author.display_name', top_n_authors=100)
    author_names = [rec['author'] for rec in top_authors]
    direct_conflicts = find_author_conflicts(reviewers_df, submitted_authors)

    # 7) Fetch institutions & get coworker conflicts:
    author_inst_df, coworker_conflicts = compute_coworker_conflicts(
        top_authors=author_names,
        submitted_author_institutions=submitted_institutions,
        email="kai.ruth.de@gmail.com",
        rate_limit=0.3
    )

    # 8) Filter out all the conflicting authors
    author_inst_df.to_csv("author_institutions.csv", index=False)

    filtered_authors, rejected_authors = filter_authors(
    top_authors,
    direct_conflicts,
    coworker_conflicts,
    submitted_authors
    )
    # 9) Save results
    reviewers_df.to_csv("reviewers_with_topic_labels.csv", index=False)
    results.to_csv("top_reviewer_matches.csv", index=False)
    author_inst_df.to_csv("author_institutions.csv", index=False)
    return {
        'reviewers_df': reviewers_df,
        'results': results,
        'domain': domain,
        'mapped_cats': mapping,
        'top_authors': top_authors,
        'author_names': author_names,
        'submitted_authors': submitted_authors,
        'submitted_abstract': manuscript_abstract,
        'submitted_institutions': submitted_institutions,
        'direct_conflicts': direct_conflicts,
        'coworker_conflicts': coworker_conflicts,
        'author_inst_df': author_inst_df,
        'filtered_authors': filtered_authors,
        'rejected_authors': rejected_authors
    }

    print("Pipeline complete. Results saved to top_authors_with_institutions.csv")

if __name__ == "__main__":
    data = main()
    top_df = pd.DataFrame(data['top_authors'])

    # # 2) get the institutions DataFrame
    # inst_df = data['author_inst_df']

    # # 3) merge on 'author'
    # merged = pd.merge(
    #     top_df,
    #     inst_df,
    #     on='author',
    #     how='left'   # use 'left' to keep all top_authors even if some lack inst info
    # )
    # merged.to_csv("top_authors_with_institutions.csv", index=False) # optional

     #You can now access all the conflict information:
    print("\nConflict Summary:")
    #print(f"- Submitted authors: {submitted_authors}")
    print(f"- Filtered list of authors: {len(data['filtered_authors'])}")
    print(f"- Rejected authors: {len(data['rejected_authors'])}")
    print("Pipeline complete.")
