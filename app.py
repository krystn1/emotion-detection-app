# app.py
import os
import io
import base64
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import numpy as np

UPLOAD_FOLDER = "static/uploads"
DB_PATH = "data/emotions.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Database helpers ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            image_path TEXT,
            dominant_emotion TEXT,
            emotions_json TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_session(name, image_path, dominant_emotion, emotions_json):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO sessions (name, image_path, dominant_emotion, emotions_json, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, image_path, dominant_emotion, emotions_json, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

# --- Utilities ---
def annotate_and_save(image_path, dominant_emotion, emotions_dict):
    # Open image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    text = f"Emotion: {dominant_emotion}"
    draw.rectangle([10, 10, 10 + 14 + len(text) * 7, 40], fill=(0,0,0,127))
    draw.text((15, 12), text, fill=(255,255,255), font=font)

    # write probabilities
    y = 50
    for k, v in emotions_dict.items():
        text = f"{k}: {v:.2f}"
        draw.text((15, y), text, fill=(255,255,255), font=font)
        y += 22

    annotated_path = image_path.replace(".","_annotated.")
    img.save(annotated_path)
    return annotated_path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, image_path, dominant_emotion, created_at FROM sessions ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Handles uploaded files from the form.
    """
    file = request.files.get("image")
    name = request.form.get("name", "Anonymous")
    if not file:
        return redirect(url_for('index'))

    # Save uploaded file
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    filename = f"{name}_{timestamp}.jpg".replace(" ", "_")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Use DeepFace pretrained emotion analyzer
    # actions=['emotion'] to limit compute to emotion attribute only.
    try:
        analysis = DeepFace.analyze(img_path=filepath, actions=['emotion'])
    except Exception as e:
        # Return a friendly error page
        return render_template("error.html", message=str(e))

    # DeepFace returns 'dominant_emotion' and 'emotion' dict (probabilities)
    if isinstance(analysis, list):
        analysis = analysis[0]

    dominant_emotion = analysis.get("dominant_emotion", "unknown")
    emotions = analysis.get("emotion", {})

    # Annotate image
    annotated_path = annotate_and_save(filepath, dominant_emotion, emotions)

    # Save results to sqlite db
    save_session(name, annotated_path, dominant_emotion, str(emotions))

    return render_template("result.html",
                           name=name,
                           image_url="/" + annotated_path.replace("\\","/"),
                           dominant_emotion=dominant_emotion,
                           emotions=emotions)

@app.route("/analyze_webcam", methods=["POST"])
def analyze_webcam():
    """
    Receives a base64 image (from client webcam capture), analyzes and returns JSON result.
    """
    data = request.get_json()
    name = data.get("name", "Anonymous")
    b64 = data.get("imageBase64")
    if not b64:
        return jsonify({"error":"No image sent"}), 400

    # decode base64 (data URI format may be used)
    header, encoded = b64.split(",", 1) if "," in b64 else ("", b64)
    img_bytes = base64.b64decode(encoded)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    filename = f"{name}_{timestamp}.jpg".replace(" ", "_")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, "wb") as f:
        f.write(img_bytes)

    try:
        analysis = DeepFace.analyze(img_path=filepath, actions=['emotion'])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if isinstance(analysis, list):
        analysis = analysis[0]

    dominant_emotion = analysis.get("dominant_emotion", "unknown")
    emotions = analysis.get("emotion", {})
    annotated_path = annotate_and_save(filepath, dominant_emotion, emotions)

    save_session(name, annotated_path, dominant_emotion, str(emotions))

    return jsonify({
        "name": name,
        "image_url": "/" + annotated_path.replace("\\","/"),
        "dominant_emotion": dominant_emotion,
        "emotions": emotions
    })

@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)