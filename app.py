import os
import cv2
import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# --- CONSTANTS AND MATH (Your Original Logic) ---
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

# Pre-load model (YOLOv8 Nano is best for Cloud Run memory limits)
model = YOLO('yolov8n.pt')

# --- USER INTERFACE (Cleaned & Corrected) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision Guard AI</title>
    <style>
        body { font-family: sans-serif; background: #121212; color: white; text-align: center; padding: 20px; }
        .card { max-width: 500px; margin: auto; background: #1e1e1e; padding: 25px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.6); }
        .btn { background: #00e676; border: none; padding: 15px; border-radius: 8px; font-weight: bold; cursor: pointer; width: 100%; font-size: 16px; margin-top: 10px; }
        .item { text-align: left; padding: 15px; margin: 15px 0; border-radius: 8px; border-left: 10px solid; }
        .STOP { background: #421212; border-color: #ff1744; }
        .WARNING { background: #3d3b0b; border-color: #ffea00; }
        .SAFE { background: #0b3d1c; border-color: #00e676; }
        .voice-btn { background: #2196f3; color: white; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Vision Guard AI</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit" class="btn">ANALYZE ENVIRONMENT</button>
        </form>

        {% if detections %}
            <button onclick="readReport()" class="btn voice-btn">🔊 READ REPORT OUT LOUD</button>
            <div id="results">
                {% for d in detections %}
                    <div class="item {{ d.status }}">
                        <strong>{{ d.label | upper }}</strong> - {{ d.pos }}<br>
                        Distance: {{ d.dist }}m | Status: {{ d.status }}
                        {% if d.status == "STOP" %}<br><span style="color:red">⚠️ ACTION: Step {{ d.step_dir }}!</span>{% endif %}
                    </div>
                {% endfor %}
            </div>
            <script>
                function readReport() {
                    const text = "{{ speech_text }}";
                    const utterance = new SpeechSynthesisUtterance(text);
                    window.speechSynthesis.speak(utterance);
                }
            </script>
        {% elif analyzed %}
            <p style="color: #00e676; margin-top:20px;">The path ahead is clear.</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    detections = []
    speech_text = "Analysis complete. "
    analyzed = False

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            analyzed = True
            img = Image.open(file).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            frame = np.array(img)[:, :, ::-1].copy()
            results = model(frame, verbose=False, conf=0.25)[0]

            for box in results.boxes:
                label = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pixel_h = y2 - y1
                center_x = (x1 + x2) / 2
                real_h = REAL_HEIGHTS.get(label, 0.6)

                distance = (real_h * FOCAL_LENGTH) / pixel_h
                if y2 > 576: distance *= 0.75 # Your ground bias

                if distance < 1.3: status = "STOP"
                elif 1.3 <= distance <= 4.0: status = "WARNING"
                else: status = "SAFE"

                if center_x < 213: pos = "to your left"
                elif center_x > 427: pos = "to your right"
                else: pos = "directly ahead"

                step_dir = "RIGHT" if center_x < 320 else "LEFT"

                detections.append({
                    "label": label, "dist": round(distance, 1),
                    "status": status, "pos": pos, "step_dir": step_dir
                })
                
                speech_text += f"{label} detected {pos} at {round(distance, 1)} meters. "
                if status == "STOP": speech_text += f"Immediate stop required! Step {step_dir}. "

            detections.sort(key=lambda x: x['dist'])

    return render_template_string(HTML_TEMPLATE, detections=detections, speech_text=speech_text, analyzed=analyzed)

if __name__ == '__main__':
    # Google Cloud Run provides the PORT environment variable. 
    # If it's missing (local testing), it defaults to 8080.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
