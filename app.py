# backend/app.py
# Final backend for CCTV project
# - processes uploaded video fully (frame-by-frame)
# - draws bounding boxes (red for emergency, green for safe)
# - counts unique physical objects (simple centroid-based tracking)
# - returns annotated .mp4 stored in backend/static/
# - max upload size: 50 MB

import os
import uuid
import math
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np

# ---------- CONFIG ----------
MODEL_FILE = "yolov8m.pt"   # ensure this file exists in backend/ (downloaded earlier)
UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
OUTPUT_WIDTH = 640  # output width (R1 choice); height keeps aspect ratio
IOU_DEDUPE_THRESH = 0.45
DISTANCE_TRACK_THRESH_RATIO = 0.08  # proportion of frame width used to decide same object

# Emergency keywords heuristic (model label substrings -> emergency)
EMERGENCY_KEYWORDS = ["knife", "gun", "rifle", "pistol", "fire", "flame", "smoke", "explosive", "bomb", "bombs"]
# You requested injured/fallen/fight/robbery to be considered; we approximate those using patterns and heuristics:
EXTRA_EMERGENCY_KEYWORDS = ["knife", "gun", "rifle", "pistol", "bomb", "fire", "smoke"]

# Safe color and emergency color in BGR
COLOR_EMERGENCY = (0, 0, 255)  # Red (B,G,R)
COLOR_SAFE = (0, 200, 0)       # Green

# Create necessary folders
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR)
app.static_folder = 'static'
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ---------- load YOLO model (ultralytics) ----------
# ---------- load YOLO GENERAL MODEL + CUSTOM MODEL ----------
model_general = None
model_custom = None

try:
    from ultralytics import YOLO

    # general YOLO model
    GENERAL_MODEL_FILE = "yolov8m.pt"
    if not os.path.exists(GENERAL_MODEL_FILE):
        raise FileNotFoundError(f"General model '{GENERAL_MODEL_FILE}' missing.")
    model_general = YOLO(GENERAL_MODEL_FILE)
    print("✅ Loaded YOLO general model:", GENERAL_MODEL_FILE)

    # your custom model
    CUSTOM_MODEL_FILE = "best.pt"
    if not os.path.exists(CUSTOM_MODEL_FILE):
        raise FileNotFoundError(f"Custom model '{CUSTOM_MODEL_FILE}' missing.")
    model_custom = YOLO(CUSTOM_MODEL_FILE)
    print("✅ Loaded custom emergency model:", CUSTOM_MODEL_FILE)

except Exception as e:
    print("❌ Model loading error:", e)

# ---------- helpers ----------
def resize_keep_aspect(frame, target_width=OUTPUT_WIDTH):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame, 1.0
    scale = target_width / float(w)
    new_h = int(h * scale)
    frame_resized = cv2.resize(frame, (target_width, new_h))
    return frame_resized, scale

def iou(boxA, boxB):
    # box: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom

def dedupe_boxes(boxes, scores, labels, iou_thresh=IOU_DEDUPE_THRESH):
    picked = []
    picked_scores = []
    picked_labels = []
    # sort by score desc
    items = sorted(list(zip(boxes, scores, labels)), key=lambda x: -x[1])
    for b, s, l in items:
        keep = True
        for pb in picked:
            if iou(b, pb) >= iou_thresh:
                keep = False
                break
        if keep:
            picked.append(b)
            picked_scores.append(s)
            picked_labels.append(l)
    return picked, picked_scores, picked_labels

def is_emergency_label(label_name):
    ln = label_name.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in ln:
            return True
    # Additional heuristics: 'person' with other dangerous contexts not handled here.
    return False

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ---------- main route ----------
@app.route("/process_video", methods=["POST"])
def process_video():
    
    if "video" not in request.files:
        return jsonify({"error": "No video file part (field name 'video')"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save upload
    uid = uuid.uuid4().hex
    filename = f"{uid}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, filename)
    try:
        file.save(upload_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save uploaded file: {str(e)}"}), 500

    # Open video
    cap = cv2.VideoCapture(upload_path)
    if not cap.isOpened():
        try:
            os.remove(upload_path)
        except: pass
        return jsonify({"error": "Cannot open uploaded video. It may be corrupted or unsupported format."}), 400

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        try:
            os.remove(upload_path)
        except: pass
        return jsonify({"error": "Uploaded video empty or unreadable."}), 400

    # Prepare writer for annotated output
    first_frame_resized, _ = resize_keep_aspect(first_frame, OUTPUT_WIDTH)
    out_h, out_w = first_frame_resized.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_filename = f"annotated_{uid}.mp4"
    out_path = os.path.join(STATIC_DIR, out_filename)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    # Tracking containers:
    # tracked_per_class: label -> list of centroids (these represent unique physical objects counted so far)
    tracked_per_class = {}
    counts_per_class = {}
    emergency_set = set()  # set of emergency labels seen
    processed_frames = 0
    total_detections = 0
    emergency_detections = 0

    # Threshold to decide same object: fraction of frame width
    dist_thresh = max(10, int(out_w * DISTANCE_TRACK_THRESH_RATIO))

    # Process each frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Resize for speed and consistency
            frame_resized, scale = resize_keep_aspect(frame, OUTPUT_WIDTH)
            # Run detection
            # Run BOTH models (general + custom)
            # ---------------------------
            # RUN BOTH MODELS TOGETHER
            # ---------------------------
            try:
                det_boxes = []
                det_scores = []
                det_labels = []

                # ---- GENERAL MODEL (YOLOv8m) ----
                if model_general is not None:
                    results_g = model_general(frame_resized)[0]
                    for box in results_g.boxes:
                        xy = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0])
                        cls_idx = int(box.cls[0])
                        label_name = model_general.names.get(cls_idx, str(cls_idx))

                        det_boxes.append(xy)
                        det_scores.append(conf)
                        det_labels.append(label_name)

                # ---- CUSTOM MODEL (best.pt) ----
                if model_custom is not None:
                    results_c = model_custom(frame_resized)[0]
                    for box in results_c.boxes:
                        xy = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0])
                        cls_idx = int(box.cls[0])
                        label_name = model_custom.names.get(cls_idx, str(cls_idx))

                        det_boxes.append(xy)
                        det_scores.append(conf)
                        det_labels.append(label_name)

            except Exception as e:
                print(f"Detection error on frame {frame_idx}: {e}")
                writer.write(frame_resized)
                continue

            # Dedupe detections
            boxes_d, scores_d, labels_d = dedupe_boxes(det_boxes, det_scores, det_labels, IOU_DEDUPE_THRESH)

            # Dedupe overlapping boxes in this frame
            #  boxes_d, scores_d, labels_d = dedupe_boxes(boxes, scores, labels, IOU_DEDUPE_THRESH)

            # For each deduped detection: draw and track
            dets_this_frame = 0
            for det_i, (b, sc, lab) in enumerate(zip(boxes_d, scores_d, labels_d)):
                total_detections += 1
                dets_this_frame += 1
                # centroid
                cx, cy = box_center(b)
                cx_i, cy_i = int(cx), int(cy)

                # check if emergency
                emergency_flag = is_emergency_label(lab) or any(k in lab.lower() for k in EXTRA_EMERGENCY_KEYWORDS)
                if emergency_flag:
                    emergency_set.add(lab.lower())
                    emergency_detections += 1

                # initialize tracking list for this class
                class_key = lab.lower()
                if class_key not in tracked_per_class:
                    tracked_per_class[class_key] = []

                # Determine if this detection is a new unique object using IOU check
                matched = False
                for tbox in tracked_per_class.get(class_key, []):
                     if iou(b, tbox) > 0.45:
                        matched = True
                        break

                if not matched:
                     tracked_per_class.setdefault(class_key, []).append(b)
                     counts_per_class[class_key] = counts_per_class.get(class_key, 0) + 1

                # draw box on frame_resized
                x1, y1, x2, y2 = [int(v) for v in b]
                color = COLOR_EMERGENCY if emergency_flag else COLOR_SAFE
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                # BIG & BOLD label (you selected)
                label_text = f"{lab}"  # label format C = only object name
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                # background rectangle for text
                tx1, ty1 = x1, max(0, y1 - th - 8)
                tx2, ty2 = x1 + tw + 8, ty1 + th + 6
                cv2.rectangle(frame_resized, (tx1, ty1), (tx2, ty2), color, -1)
                cv2.putText(frame_resized, label_text, (x1 + 4, ty1 + th + -2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # timestamp
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame_resized, ts, (8, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            writer.write(frame_resized)
            processed_frames += 1

    finally:
         cap.release()
         writer.release()
         try:
             os.remove(upload_path)
         except:
              pass

# ------------------------------------
# THIS MUST BE OUTSIDE FINALLY BLOCK
# ------------------------------------
    summary = {
        "annotated_url": f"http://127.0.0.1:5000/static/{out_filename}",
        "processed_frames": int(processed_frames),
        "total_detections": int(total_detections),
        "emergency_detections": int(emergency_detections),
        "emergencies": list(emergency_set),
        "counts_per_class": counts_per_class
    }

    return jsonify(summary), 200

# Serve static annotated file
@app.route(f"/{STATIC_DIR}/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename,as_attachment=True)
# ------------------------------------
# CLEANUP WHEN SERVER STOPS
# ------------------------------------
import atexit
import glob

def cleanup_temp_files():
    print("🧹 Cleaning temporary files...")

    # delete annotated videos
    for file in glob.glob(os.path.join(STATIC_DIR, "annotated_*.mp4")):
        try:
            os.remove(file)
            print("Deleted:", file)
        except:
            pass

    # delete uploaded raw videos
    for file in glob.glob(os.path.join(UPLOAD_DIR, "*")):
        try:
            os.remove(file)
            print("Deleted:", file)
        except:
            pass

atexit.register(cleanup_temp_files)

# ------------------------------------
# START SERVER (DO NOT TOUCH THIS)
# ------------------------------------

if __name__ == "__main__":
    print("Starting backend server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)


