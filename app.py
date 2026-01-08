import os
import cv2
import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# --- CONSTANTS AND MATH (Identical to your logic) ---
IMAGE_SIZE = 640
VFOV_DEG = 60
FOCAL_LENGTH = (IMAGE_SIZE / 2) / np.tan(np.deg2rad(VFOV_DEG / 2))

REAL_HEIGHTS = {
    "person": 1.70, "child": 1.20, "dog": 0.50, "cat": 0.25, "bird": 0.15,
    "couch": 0.85, "sofa": 0.85, "chair": 0.90, "armchair": 1.00,
    "bed": 0.60, "dining table": 0.75, "desk": 0.75, "coffee table": 0.45,
    "side table": 0.55, "shelf": 1.60, "bookcase": 1.80, "wardrobe": 2.00,
    "cabinet": 0.80, "stool": 0.50, "bench": 0.45, "toilet": 0.45,
    "tv": 0.55, "monitor": 0.45, "laptop": 0.25, "refrigerator": 1.75,
    "microwave": 0.35, "oven": 0.85, "stove": 0.85, "sink": 0.85,
    "washing machine": 0.85, "vacuum cleaner": 1.00, "clock": 0.30,
    "vase": 0.35, "potted plant": 0.70, "lamp": 0.50, "chandelier": 0.70,
    "trash can": 0.50, "bucket": 0.35, "backpack": 0.50, "handbag": 0.30,
    "car": 1.50, "suv": 1.70, "van": 2.00, "truck": 3.00, "bus": 3.20,
    "bicycle": 1.00, "motorcycle": 1.10, "scooter": 1.00,
    "fire hydrant": 0.80, "stop sign": 2.50, "traffic light": 4.00,
    "parking meter": 1.30, "bench (outdoor)": 0.80, "mailbox": 1.20,
    "street light": 6.00, "tree": 5.00, "fence": 1.20,
    "door": 2.05, "window": 1.20, "stairs": 1.00, "elevator": 2.20
}

# Load YOLO model
model = YOLO('yolov8n.pt')

# --- CLEANED HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Vision Guard AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: white; text-align: center; padding: 20px; }
        .container { max-width: 600px; margin: auto; background: #1e1e1e; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        .upload-box { border: 2px dashed #444; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        input[type="file"] { margin: 15px 0; }
        button { background: #00e676; border: none; padding: 12px 25px; border-radius: 8px; font-weight: bold; cursor: pointer; color: #121212; width: 100%; font-size: 16px; }
        button:hover { background: #00c853; }
        .results-container { margin-top: 30px; text-align: left; }
        .item { padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 8px solid; }
        .STOP { background: #3d0b13; border-color: #ff1744; }
        .WARNING { background: #3d3b0b; border-color: #ffea00; color: #fff; }
        .SAFE { background: #0b3d1c; border-color: #00e676; }
        .action-alert { font-weight: bold; display: block; margin-top: 10px; color: #ff1744; border-top: 1px solid rgba(255,23,68,0.3); padding-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Vision Guard AI</h1>
        <p>Assistant for the Visually Impaired</p>
        
        <div class="upload-box">
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required><br>
                <button type="submit">ANALYZE ENVIRONMENT</button>
            </form>
        </div>

        {% if detections %}
            <div class="results-container">
                <h2 style="text-align: center;">Safety Report</h2>
                {% for d in detections %}
                    <div class="item {{ d.status }}">
                        <strong>{{ d.label | upper }}</strong> — {{ d.pos }}<br>
                        Distance: {{ d.dist }} meters
                        {% if d.status == "STOP" %}
                            <span class="action-alert">⚠️ ACTION: Step {{ d.step_dir }} immediately!</span>
                        {% endif %}
                    </div>
                {% endfor %}
            </div>
        {% elif analyzed %}
            <p style="color: #00e676;">No objects detected. The path is clear.</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    detections = []
    analyzed = False
    
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            analyzed = True
            # Image Processing
            img = Image.open(file).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            frame = np.array(img)[:, :, ::-1].copy()
            results = model(frame, verbose=False, conf=0.20)[0]

            print(f"\n--- GLOBAL SAFETY REPORT (Detected {len(results.boxes)} objects) ---")

            for box in results.boxes:
                label = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pixel_h = y2 - y1
                center_x = (x1 + x2) / 2
                real_h = REAL_HEIGHTS.get(label, 0.6)
                
                # Math Logic
                distance = (real_h * FOCAL_LENGTH) / pixel_h
                if y2 > 576: distance *= 0.75

                # Safety Status
                if distance < 1.3: status = "STOP"
                elif 1.3 <= distance <= 4.0: status = "WARNING"
                else: status = "SAFE"

                # Positioning
                if center_x < 213: pos = "to your LEFT"
                elif center_x > 427: pos = "to your RIGHT"
                else: pos = "DIRECTLY AHEAD"

                step_dir = "RIGHT" if center_x < 320 else "LEFT"

                # Log to Cloud Console
                print(f"[{status}] {label} {pos} at {distance:.1f} meters.")

                detections.append({
                    "label": label,
                    "dist": round(distance, 1),
                    "status": status,
                    "pos": pos,
                    "step_dir": step_dir
                })

            # Sort by distance
            detections.sort(key=lambda x: x['dist'])

    return render_template_string(HTML_TEMPLATE, detections=detections, analyzed=analyzed)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
