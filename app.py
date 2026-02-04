#!pip install ngrok
#!pip install flask flask-cors rdkit xgboost
#!pip install pyngrok

"""
XGBoost Toxicity Prediction - Flask(google colab)

Multi-label XGBoost toxicity model trained on Tox21.

Deployment choices:
- Model: XGBoost (Morgan fingerprints)
- Decision threshold: 0.7
- Macro-F1 (test, 0.7): ~0.44
- Macro-ROC-AUC (test): ~0.78
"""

# !pip install ngrok
# !pip install flask flask-cors rdkit xgboost
# !pip install pyngrok

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict

# Import from existing modules instead of redefining
from src.toxicity.data.fingerprint_data_featurization import generate_morgan_fingerprint, generate_descriptors
from src.toxicity.models.xgboost_multilabel import MultiLabelXGBoost, TOX21_ASSAYS

# cleaning (salt removal + canonicalization + size filter)
from src.toxicity.data.cleaning import clean_molecule

# OPTIONAL: NGROK (google colab)
USE_NGROK = True  # False if running locally

if USE_NGROK:
    try:
        from pyngrok import ngrok
    except ImportError:
        import os
        os.system("pip -q install pyngrok")
        from pyngrok import ngrok
    ngrok.set_auth_token(ngrok token)

# LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xgboost-toxicity-api")

# CONFIGURATION
MODEL_PATH = Path("models/xgboost_baseline")
DEFAULT_THRESHOLD = 0.7
FP_RADIUS = 2
FP_NBITS = 2048

#cleaning helper 
def clean_smiles_or_raise(smiles: str) -> str:
    """
    Clean SMILES using your ordered pipeline:
    validate -> salt removal -> canonicalize -> size filter
    Returns cleaned SMILES or raises ValueError with reason.
    """
    cleaned, info = clean_molecule(
        smiles,
        canonicalize=True,
        remove_salt=True,
        min_atoms=3,
        max_atoms=150,
        verbose=False
    )
    if cleaned is None:
        raise ValueError(info.get("reason", "SMILES cleaning failed"))
    return cleaned

# FEATURE EXTRACTION

def smiles_to_fingerprint(smiles: str) -> np.ndarray:
    """Convert a SMILES string to a Morgan fingerprint."""
    smiles = clean_smiles_or_raise(smiles)  
    fp = generate_morgan_fingerprint(smiles, radius=FP_RADIUS, n_bits=FP_NBITS)
    if fp is None:
        raise ValueError("Invalid SMILES string")
    return fp.astype(np.float32)


def compute_molecular_descriptors(smiles: str) -> Dict[str, float]:
    """Compute basic molecular descriptors for interpretability."""
    smiles = clean_smiles_or_raise(smiles)  

    descriptor_names = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
                       'NumRotatableBonds', 'NumAromaticRings', 'TPSA']

    desc = generate_descriptors(smiles, descriptors=descriptor_names)
    if desc is None:
        return {}

    return {
        "molecular_weight": desc[0],
        "log_p": desc[1],
        "num_h_donors": desc[2],
        "num_h_acceptors": desc[3],
        "num_rotatable_bonds": desc[4],
        "num_aromatic_rings": desc[5],
        "tpsa": desc[6],
    }


# MODEL WRAPPER

class XGBoostPredictor:
    """Wrapper around per-assay XGBoost binary classifiers."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.multilabel_model = MultiLabelXGBoost.load(str(model_dir))
        logger.info("Loaded %d / %d assay models",
                   len(self.multilabel_model.models), len(TOX21_ASSAYS))

    def predict(self, fingerprint: np.ndarray, threshold: float) -> Dict[str, Dict]:
        """Predict toxicity probabilities and labels for all assays."""
        x = fingerprint.reshape(1, -1)

        # Get probabilities from multilabel model
        probas = self.multilabel_model.predict_proba(x)[0]

        predictions = {}
        probabilities = {}

        for i, assay in enumerate(TOX21_ASSAYS):
            prob = float(probas[i])
            if np.isnan(prob):
                predictions[assay] = None
                probabilities[assay] = None
            else:
                probabilities[assay] = prob
                predictions[assay] = "TOXIC" if prob >= threshold else "NON-TOXIC"

        return {"predictions": predictions, "probabilities": probabilities}


# FLASK APP

app = Flask(__name__)
CORS(app)

try:
    predictor = XGBoostPredictor(MODEL_PATH)
except Exception as e:
    logger.error("Failed to load models: %s", e)
    predictor = None


# HTML UI

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>XGBoost Toxicity Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f5f7fb; padding: 30px; }
    .container { max-width: 1000px; margin: auto; background: white;
      padding: 30px; border-radius: 12px; }
    textarea { width: 100%; font-family: monospace; font-size: 15px; padding: 12px; }
    button { padding: 12px 20px; font-size: 16px; margin-top: 10px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; }
    .toxic { background: #fff3f3; border-color: #d9534f; }
    .non-toxic { background: #f3fff6; border-color: #5cb85c; }
  </style>
</head>
<body>
  <div class="container">
    <h1>XGBoost Toxicity Predictor</h1>
    <p>Decision threshold: 0.7</p>

    <textarea id="smiles" rows="3" placeholder="Enter SMILES string"></textarea>
    <button onclick="predict()">Predict</button>

    <div id="results" class="grid" style="margin-top:20px;"></div>
  </div>

  <script>
    async function predict() {
      const smiles = document.getElementById("smiles").value;

      const res = await fetch("/api/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({smiles: smiles})
      });

      const data = await res.json();

      const grid = document.getElementById("results");
      grid.innerHTML = "";

      if (data.error) {
        const div = document.createElement("div");
        div.className = "card toxic";
        div.innerHTML = "<strong>Error</strong><br>" + data.error;
        grid.appendChild(div);
        return;
      }

      for (const assay in data.predictions) {
        const pred = data.predictions[assay];
        const prob = data.probabilities[assay];

        if (pred === null) continue;

        const div = document.createElement("div");
        div.className = "card " + (pred === "TOXIC" ? "toxic" : "non-toxic");
        div.innerHTML = "<strong>" + assay + "</strong><br>" +
                        pred + "<br>" +
                        "Probability: " + (prob * 100).toFixed(1) + "%";
        grid.appendChild(div);
      }
    }
  </script>
</body>
</html>
"""


# ROUTES

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True) or {}
    smiles = str(data.get("smiles", "")).strip()
    threshold = float(data.get("threshold", DEFAULT_THRESHOLD))

    if not smiles:
        return jsonify({"error": "SMILES string required"}), 400

    start = time.time()

    try:
        smiles_clean = clean_smiles_or_raise(smiles)

        fingerprint = smiles_to_fingerprint(smiles_clean)
        result = predictor.predict(fingerprint, threshold)
        descriptors = compute_molecular_descriptors(smiles_clean)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return jsonify({
        "smiles": smiles,
        "smiles_clean": smiles_clean,  
        "predictions": result["predictions"],
        "probabilities": result["probabilities"],
        "molecular_descriptors": descriptors,
        "threshold": threshold,
        "inference_time_ms": elapsed_ms,
        "model": "XGBoost"
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": predictor is not None,
        "n_models": len(predictor.multilabel_model.models) if predictor else 0
    })


# ENTRY POINT (WITH NGROK HOOK)

if __name__ == "__main__":
    print("=" * 80)
    print("XGBoost Toxicity Prediction Server")
    print("Threshold:", DEFAULT_THRESHOLD)
    print("=" * 80)

    if USE_NGROK:
        try:
            ngrok.kill()
        except Exception:
            pass

        public_url = ngrok.connect(5000)
        print("Public URL:", public_url)

    # Run Flask
    app.run(host="0.0.0.0", port=5000, debug=False)
