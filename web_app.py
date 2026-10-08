from pathlib import Path
from datetime import date, timedelta
import json
import subprocess
import time
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, Response, abort, redirect, render_template, request, send_from_directory, stream_with_context, url_for
from werkzeug.utils import secure_filename

from detection import AccidentDetectionModel
from report_data import (
    get_area_period_report,
    get_incident_report,
    get_report_areas,
    get_sample_vehicle_numbers,
    initialize_reports_database,
    normalize_vehicle_number,
    search_vehicle_incidents,
)

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
analysis_jobs = {}
initialize_reports_database()


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
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
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
                prediction_started = time.perf_counter()
                prediction, probabilities = model.predict_accident(image[np.newaxis, :, :])
                last_prediction = prediction
                last_probability = float(np.max(probabilities))
                highest_probability = max(highest_probability, last_probability)
                accident_detected = prediction == "Accident" and last_probability >= ACCIDENT_THRESHOLD
                if accident_detected:
                    accident_frames += 1
                yield {
                    "type": "progress",
                    "frame": frame_number,
                    "total_frames": total_frames,
                    "timestamp_seconds": round((frame_number - 1) / fps, 3),
                    "prediction": last_prediction,
                    "confidence": round(last_probability * 100, 1),
                    "accident_detected": accident_detected,
                    "elapsed_ms": round((time.perf_counter() - prediction_started) * 1000),
                }

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

    yield {
        "type": "complete",
        "frames": frame_number,
        "accident_frames": accident_frames,
        "highest_probability": round(highest_probability * 100, 1),
    }


@app.route("/", methods=["GET"])
def index():
    query = request.args.get("vehicle", "").strip()
    searched = bool(query)
    results = search_vehicle_incidents(query) if searched else []
    return render_template(
        "index.html",
        query=query,
        normalized=normalize_vehicle_number(query),
        searched=searched,
        results=results,
        total_deaths=sum(row["deaths"] for row in results),
        samples=get_sample_vehicle_numbers(),
    )


@app.route("/analyze-video", methods=["GET"])
def video_page():
    return render_template("video.html", error=request.args.get("error"))


def report_filters():
    default_end = date.today()
    default_start = default_end - timedelta(days=364)
    start_value = request.args.get("start", default_start.isoformat())
    end_value = request.args.get("end", default_end.isoformat())
    area_value = request.args.get("area", "")

    try:
        start_date = date.fromisoformat(start_value)
        end_date = date.fromisoformat(end_value)
        if start_date > end_date:
            raise ValueError
    except ValueError:
        start_date = default_start
        end_date = default_end
        start_value = start_date.isoformat()
        end_value = end_date.isoformat()

    area_ids = {str(area["id"]) for area in get_report_areas()}
    selected_area = area_value if area_value in area_ids else ""
    end_exclusive = end_date + timedelta(days=1)
    filters = {
        "start": start_value,
        "end": end_value,
        "area": selected_area,
        "start_datetime": f"{start_date.isoformat()} 00:00:00",
        "end_exclusive": f"{end_exclusive.isoformat()} 00:00:00",
    }
    return filters


@app.route("/reports")
def reports():
    return redirect(url_for("area_period_report"))


@app.route("/reports/areas")
def area_period_report():
    filters = report_filters()
    area_id = int(filters["area"]) if filters["area"] else None
    rows = get_area_period_report(filters["start_datetime"], filters["end_exclusive"], area_id)
    total_accidents = sum(row["accident_count"] for row in rows)
    total_deaths = sum(row["deaths"] for row in rows)
    max_accidents = max((row["accident_count"] for row in rows), default=0)
    return render_template(
        "report_areas.html",
        areas=get_report_areas(),
        rows=rows,
        filters=filters,
        total_accidents=total_accidents,
        total_deaths=total_deaths,
        max_accidents=max_accidents,
    )


@app.route("/reports/incidents")
def incident_report():
    filters = report_filters()
    area_id = int(filters["area"]) if filters["area"] else None
    rows = get_incident_report(filters["start_datetime"], filters["end_exclusive"], area_id)
    return render_template(
        "report_incidents.html",
        areas=get_report_areas(),
        rows=rows,
        filters=filters,
        total_deaths=sum(row["deaths"] for row in rows),
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded_file = request.files.get("video")
    if uploaded_file is None or uploaded_file.filename == "":
        return redirect(url_for("video_page", error="Choose a video file first."))
    if not allowed_file(uploaded_file.filename):
        return redirect(url_for("video_page", error="Supported formats: MP4, AVI, MOV, MKV, and WEBM."))

    job_id = uuid4().hex
    filename = secure_filename(uploaded_file.filename)
    input_path = UPLOAD_DIR / f"{job_id}_{filename}"
    output_filename = f"{job_id}_processed.mp4"
    output_path = RESULT_DIR / output_filename
    uploaded_file.save(input_path)
    analysis_jobs[job_id] = {
        "input_path": input_path,
        "output_path": output_path,
        "output_filename": output_filename,
        "status": "pending",
        "summary": None,
        "error": None,
    }
    return redirect(url_for("analysis_page", job_id=job_id))


@app.route("/analysis/<job_id>")
def analysis_page(job_id):
    if job_id not in analysis_jobs:
        abort(404)
    return render_template("analysis.html", job_id=job_id)


@app.route("/uploads/<job_id>")
def source_video(job_id):
    job = analysis_jobs.get(job_id)
    if job is None or not job["input_path"].is_file():
        abort(404)
    return send_from_directory(UPLOAD_DIR, job["input_path"].name, as_attachment=False)


@app.route("/analysis/<job_id>/events")
def analysis_events(job_id):
    job = analysis_jobs.get(job_id)
    if job is None:
        abort(404)
    if job["status"] == "complete":
        return Response(json.dumps(job["summary"]) + "\n", mimetype="application/x-ndjson")
    if job["status"] == "error":
        return Response(json.dumps({"type": "error", "message": job["error"]}) + "\n", mimetype="application/x-ndjson")
    if job["status"] != "pending":
        return {"error": "This analysis is already in progress."}, 409

    job["status"] = "processing"

    @stream_with_context
    def events():
        try:
            for event in process_video(job["input_path"], job["output_path"]):
                if event["type"] == "complete":
                    event["video"] = url_for("results", filename=job["output_filename"])
                    event["download"] = url_for("download", filename=job["output_filename"])
                    job["summary"] = event
                    job["status"] = "complete"
                yield json.dumps(event) + "\n"
        except (OSError, ValueError) as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        finally:
            job["input_path"].unlink(missing_ok=True)

    return Response(events(), mimetype="application/x-ndjson")


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
