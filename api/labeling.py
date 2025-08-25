import re
import pandas as pd

from .models import get_topic_classifier


def load_topic_classifier(model_name = "OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract"):
    return get_topic_classifier(model_name)


def label_submitted_abstract(abstract, classifier):
    result = classifier(abstract, truncation=True)
    if isinstance(result, list) and result and isinstance(result[0], dict) and 'label' in result[0]:
        return result[0]['label']
    if isinstance(result, dict) and 'label' in result:
        return result['label']
    raise ValueError(f"Unexpected classifier output: {result}")


def label_all_abstracts(df, classifier, text_col='abstract', batch_size=16):
    texts = df[text_col].fillna('').tolist()
    labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results = classifier(batch, truncation=True)
        for res in results:
            if isinstance(res, list) and res and isinstance(res[0], dict) and 'label' in res[0]:
                labels.append(res[0]['label'])
            elif isinstance(res, dict) and 'label' in res:
                labels.append(res['label'])
            else:
                labels.append('')
    new_df = df.copy()
    new_df['topic_label'] = labels
    return new_df


def extract_topic_id_and_name(topic_label):
    m = re.match(r'^\s*(\d+)\s*:\s*(.+)$', topic_label)
    if m:
        topic_id = int(m.group(1))
        clean_name = m.group(2).strip()
    else:
        topic_id = None
        clean_name = topic_label.strip()
    return topic_id, clean_name


