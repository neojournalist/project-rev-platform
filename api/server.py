from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import sys
import os


def load_env():
    """Load simple KEY=VALUE pairs from .env files into environment variables."""
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '.env'),  # project root
        os.path.join(os.path.dirname(__file__), '.env'),        # api/.env
        '.env',                                                # cwd
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, 'r') as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            except Exception:
                pass

# Load environment on startup
load_env()

# Import from the api package to support relative imports inside submodules
from api.functions import load_reviewers, load_topic_classifier, label_submitted_abstract, map_topic_by_id
from api.api import run_expertise_pipeline

app = Flask(__name__)
#CORS(app)  # Allow requests from React frontend
CORS(app,resources={
        r"/run_pipeline": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    }, supports_credentials=True)



@app.route("/run_pipeline", methods=["POST", "OPTIONS"])
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
            mapping_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "topic_mapping.csv"))
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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
