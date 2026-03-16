from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import subprocess
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/model-1.keras"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Audio backend running successfully"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    cmd = [
        "python",
        "command_line_model_tester.py",
        "--model",
        MODEL_PATH,
        "--clips",
        file_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return jsonify({
            "error": "Model execution failed",
            "details": result.stderr
        }), 500

    return jsonify({
        "status": "success",
        "model_output": result.stdout.strip()
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
