import modal
import os
import pandas as pd
app = modal.App("expertise-api")

# ➊ Pick or build an image with everything your pipeline needs
image = (
    modal.Image.debian_slim()
    .pip_install(             # add any other PyPI deps here
        "flask",
        "flask_cors",
        # put heavy ML libs etc. in this list too
        "pandas",
        "numpy",
        "scikit-learn",
        "torch",
        "sentence-transformers"
    )
    # If run_expertise_pipeline lives in your repo, copy it in:
    .add_local_dir(".", "/root", ignore=["venv", "__pycache__"])
)
credentials = modal.Secret.from_name("my-credentials")


@app.function(image=image, secrets=[credentials])  
@modal.concurrent(max_inputs=100)    # optional but recommended
@modal.wsgi_app()    # label → …--pipeline.modal.run
def flask_server():
    """Return a ready-to-serve Flask WSGI app."""
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from api import run_expertise_pipeline     # local import inside container
    import pandas as pd
    from functions import load_reviewers, load_topic_classifier, label_submitted_abstract, map_topic_by_id
    from api import run_expertise_pipeline

    app = Flask(__name__)
    #CORS(app)  # Allow requests from React frontend
    CORS(
        app,
        resources={r"/*": {"origins": ["https://peer-match-wheat.vercel.app"]}},
        supports_credentials=True,        # only if your fetch sends credentials
        allow_headers=["Content-Type"],   # let pre-flight know JSON is OK
        methods=["POST", "OPTIONS"],      # what the browser may send
    )

    @app.post("/run_pipeline")
    def run_pipeline():
        # Flask-CORS will auto-handle OPTIONS
        if request.method == "OPTIONS":
            # short-circuit preflight
            return jsonify({}), 200
        
        try:
            data = request.get_json()
            manuscript = data.get("manuscript", "")
            authors = data.get("authors", "")
            institutions = data.get("institutions", "")
            domain = data.get("domain", None)
            authors_list = [a.strip() for a in authors.split("\n") if a.strip()]
            institutions_list = [i.strip() for i in institutions.split("\n") if i.strip()]
            
            # If no domain is specified, just return classification data
            if domain is None:
                print("🔍 Running classification only...")
                classifier = load_topic_classifier()
                submitted_label = label_submitted_abstract(manuscript, classifier)
                mapping_df = pd.read_csv("topic_mapping.csv")
                mapping = map_topic_by_id(mapping_df, submitted_label)
                
                # Convert any NaN values to None for JSON serialization
                if mapping:
                    for key, value in mapping.items():
                        # Convert numpy types to Python native types for JSON serialization
                        if hasattr(value, 'item'):  # numpy scalar types
                            mapping[key] = value.item()
                        elif isinstance(value, (list, tuple)):
                            # For lists/arrays, check if any element is NaN or numpy type
                            new_list = []
                            for item in value:
                                if pd.isna(item):
                                    new_list.append(None)
                                elif hasattr(item, 'item'):  # numpy scalar types
                                    new_list.append(item.item())
                                else:
                                    new_list.append(item)
                            mapping[key] = new_list
                        elif pd.isna(value):
                            mapping[key] = None
                
                result = {
                    "label": submitted_label,
                    "mapping": mapping
                }
                print(f"✅ Classification result: {result}")
                return jsonify(result)
            else:
                # Run full pipeline when domain is specified
                print("🚀 Running full pipeline with domain:", domain)
                results = run_expertise_pipeline(manuscript, authors_list, institutions_list, domain=domain)
                return jsonify(results)
                
        except Exception as e:
            import traceback
            print(f"❌ Error in pipeline: {str(e)}")
            print(f"❌ Full traceback: {traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    return app
