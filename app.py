import os
import cv2
import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# --- YOUR ORIGINAL CONSTANTS AND MATH ---
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

model = YOLO('yolov8n.pt')

# HTML for the web browser
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Vision Guard AI</title>
    <style>
        body { font-family: sans-serif; background: #121212; color: white; text-align: center; padding: 20px; }
        .item { padding: 15px; margin: 10px; border-radius: 10px; text-align: left; border-left: 10px solid; }
        .STOP { background: #3d0b13; border-color: #ff1744; }
        .WARNING { background: #3d3b0b; border-color: #ffea00; }
        .SAFE { background: #0b3d1c; border-color: #00e676; }
    </style>
</head>
<body>
    <h1>Vision Guard AI</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required><br><br>
        <button type="submit" style="padding:10px 20px;">ANALYZE ENVIRONMENT</button>
    </form>
    {% if detections %}
        <div style="max-width:600px; margin:auto; margin-top:30px;">
            {% for d in detections %}
                <div class="item {{ d.status }}">
                    <strong>{{ d.label }}</strong> - {{ d.pos }} at {{ d.dist }}m
                </div>
            {% endfor %}
        </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    detections = []
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            img = Image.open(file).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            frame = np.array(img)[:, :, ::-1].copy()
            results = model(frame, verbose=False, conf=0.20)[0]

            # --- PRINTING TO CLOUD LOGS ---
            print(f"\n--- GLOBAL SAFETY REPORT ---")

            for box in results.boxes:
                label = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pixel_h = y2 - y1
                center_x = (x1 + x2) / 2
                real_h = REAL_HEIGHTS.get(label, 0.6)
                distance = (real_h * FOCAL_LENGTH) / pixel_h
                if y2 > 576: distance *= 0.75

                if distance < 1.3: status = "STOP"
                elif 1.3 <= distance <= 4.0: status = "WARNING"
                else: status = "SAFE"

                if center_x < 213: pos = "to your LEFT"
                elif center_x > 427: pos = "to your RIGHT"
                else: pos = "DIRECTLY AHEAD"

                # This prints to your Google Cloud Logs
                print(f"[{status}] {label} {pos} at {distance:.1f} meters.")

                detections.append({
                    "label": label.upper(), "dist": round(distance, 1),
                    "status": status, "pos": pos
                })

            detections.sort(key=lambda x: x['dist'])

    return render_template_string(HTML_TEMPLATE, detections=detections)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
