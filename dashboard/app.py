"""
Simple Flask dashboard that streams annotated frames from a *video* source.
By default, it does NOT connect to AirSim (keeps it lightweight for judges).
You can adapt it later to share frames from the AirSim loop via a queue.
"""
from flask import Flask, Response, render_template_string
import cv2
from ai.detector import PersonDetector

HTML = """
<!doctype html>
<html>
  <head>
    <title>AI Drone Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; padding: 2rem; background: #0b1220; color: #e8f0fe; }
      .card { max-width: 960px; margin: 0 auto; background: #121a2d; border-radius: 16px; padding: 1rem 1rem 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
      h1 { margin: 0 0 1rem 0; font-weight: 700; letter-spacing: 0.3px; }
      .meta { opacity: 0.8; margin-bottom: 1rem; }
      img { width: 100%; border-radius: 12px; display: block; }
      footer { opacity: 0.6; font-size: 0.9rem; margin-top: 1rem; }
      code { background: #0d1325; padding: 0.2rem 0.4rem; border-radius: 6px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>AI Drone Dashboard</h1>
      <div class="meta">Live person detection (video source)</div>
      <img src="{{ url_for('video_feed') }}" alt="Live stream"/>
      <footer>Tip: Replace video source with AirSim frames by sending frames to a queue.</footer>
    </div>
  </body>
</html>
"""

app = Flask(__name__)

def gen_frames():
    det = PersonDetector(conf=0.5)
    cap = cv2.VideoCapture("data/sample.mp4")
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # fallback to webcam

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            continue
        annotated, _ = det.detect(frame)
        ret, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + bytearray(buf) + b"\r\n")

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
