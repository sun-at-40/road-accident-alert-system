from pathlib import Path
import subprocess
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, abort, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from detection import AccidentDetectionModel

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "web_uploads"
RESULT_DIR = BASE_DIR / "web_results"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
DETECTION_INTERVAL = 3
ACCIDENT_THRESHOLD = 0.5

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
model = AccidentDetectionModel(str(BASE_DIR / "model.json"), str(BASE_DIR / "model_weights.keras"))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video(input_path, output_path):
    video = cv2.VideoCapture(str(input_path))
    if not video.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        video.release()
        raise ValueError("The processed video could not be created.")

    frame_number = 0
    accident_frames = 0
    highest_probability = 0.0
    last_prediction = "No Accident"
    last_probability = 0.0

    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            frame_number += 1
            if frame_number == 1 or frame_number % DETECTION_INTERVAL == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(rgb_frame, (250, 250))
                prediction, probabilities = model.predict_accident(image[np.newaxis, :, :])
                last_prediction = prediction
                last_probability = float(np.max(probabilities))
                highest_probability = max(highest_probability, last_probability)
                if prediction == "Accident" and last_probability >= ACCIDENT_THRESHOLD:
                    accident_frames += 1

            label = f"{last_prediction} {last_probability * 100:.1f}%"
            color = (0, 0, 255) if last_prediction == "Accident" else (0, 150, 0)
            cv2.rectangle(frame, (0, 0), (300, 42), (20, 20, 20), -1)
            cv2.putText(frame, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            writer.write(frame)
    finally:
        video.release()
        writer.release()

    browser_path = output_path.with_name(f"{output_path.stem}_browser.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(output_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an", str(browser_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        output_path.unlink(missing_ok=True)
        browser_path.replace(output_path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        browser_path.unlink(missing_ok=True)

    return {
        "frames": frame_number,
        "accident_frames": accident_frames,
        "highest_probability": round(highest_probability * 100, 1),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("video")
        if uploaded_file is None or uploaded_file.filename == "":
            error = "Choose a video file first."
        elif not allowed_file(uploaded_file.filename):
            error = "Supported formats: MP4, AVI, MOV, MKV, and WEBM."
        else:
            job_id = uuid4().hex
            filename = secure_filename(uploaded_file.filename)
            input_path = UPLOAD_DIR / f"{job_id}_{filename}"
            output_filename = f"{job_id}_processed.mp4"
            output_path = RESULT_DIR / output_filename
            uploaded_file.save(input_path)
            try:
                summary = process_video(input_path, output_path)
                result = {"video": output_filename, **summary}
            except (OSError, ValueError) as exc:
                error = str(exc)
            finally:
                input_path.unlink(missing_ok=True)

    return render_template("index.html", result=result, error=error)


@app.route("/results/<path:filename>")
def results(filename):
    if not (RESULT_DIR / filename).is_file():
        abort(404)
    return send_from_directory(RESULT_DIR, filename, as_attachment=False)


@app.route("/download/<path:filename>")
def download(filename):
    if not (RESULT_DIR / filename).is_file():
        abort(404)
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
