from functools import lru_cache
import torch


@lru_cache(maxsize=1)
def get_specter_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "allenai/specter2_base",
        model_kwargs={'adapter_name': 'allenai/specter2'}
    )
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


@lru_cache(maxsize=1)
def get_topic_classifier(model_name: str = "OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract"):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )


