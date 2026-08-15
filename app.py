"""
Web dashboard for the Smart Garage Parking Assistant.
Serves a live webcam feed with detection boxes drawn on it, plus a
real-time alignment/distance gauge — viewable from any browser on the
same WiFi network (phone, Mac, etc).

Run with: python3 app.py
Then visit: http://<pi-ip>:5000
"""

import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string

from camera import AlignmentCamera
from distance_sensor import UltrasonicDistance
from fusion import ParkingFusion

STOP_DISTANCE_CM = 20
CENTER_TOLERANCE_PX = 40
LOOP_DELAY_S = 0.02  # small yield only; detection itself (~0.2-0.3s) sets the real pace

# YOLO inference size in px; lower = faster but less accurate on small/far
# objects. Combined with ROI_RIGHT_FRAC below (which crops the wide 16:9
# frame toward square before this resize), 288 lands well ahead of the old
# full-frame 640 default on both speed and accuracy.
DETECTION_IMGSZ = 288

# Only the left ROI_RIGHT_FRAC fraction of the frame (from the left edge) is
# captured/analyzed -- the rest of the garage (e.g. the neighboring car's
# spot) is cropped away before detection ever sees it, and never appears in
# the video feed either. Tune by running `python3 camera.py`, which saves
# roi_snapshot.jpg so you can check the crop line lands where the left spot
# ends. Keep this in sync with main.py.
ROI_RIGHT_FRAC = 0.55

# Calibration offset (px, ROI-frame units) so "centered in the spot" reads
# as 0 even if the camera itself is mounted off-axis. Run `python3 camera.py`
# with the car parked exactly where it should be and set this to the offset
# it reports. Keep this in sync with main.py.
CENTER_OFFSET_PX = 0

app = Flask(__name__)

cam = AlignmentCamera(imgsz=DETECTION_IMGSZ, center_offset_px=CENTER_OFFSET_PX,
                       roi_right_frac=ROI_RIGHT_FRAC)
ultrasonic = UltrasonicDistance()
fusion = ParkingFusion(stop_distance_cm=STOP_DISTANCE_CM, center_tolerance_px=CENTER_TOLERANCE_PX)

# shared state, updated by the background loop, read by Flask routes
state_lock = threading.Lock()
latest = {
    "frame": None,
    "offset_px": 0,
    "frame_width": cam.frame_width,
    "distance_cm": None,
    "car_detected": False,
    "guidance": "STARTING",
}


def sensor_loop():
    while True:
        ret, raw_frame = cam.read_frame()
        offset_px, box_height_px, car_detected = cam.detect(raw_frame) if ret else (0, 0, False)
        distance_cm = ultrasonic.get_distance_cm()
        guidance = fusion.get_guidance(offset_px, car_detected, distance_cm)

        annotated = raw_frame if ret else None
        if ret and car_detected:
            # offset_px is relative to the calibrated garage center, not the
            # camera's raw geometric center, so the reference line has to
            # shift by the same calibration offset to match.
            garage_center_x = int(cam.frame_width / 2 + cam.center_offset_px)
            cx = int(garage_center_x + offset_px)
            cv2.line(annotated, (garage_center_x, 0), (garage_center_x, 40), (0, 255, 0), 2)
            cv2.circle(annotated, (cx, 20), 8, (0, 0, 255), -1)
            cv2.putText(annotated, guidance, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        with state_lock:
            latest["frame"] = annotated
            latest["offset_px"] = offset_px
            latest["distance_cm"] = distance_cm
            latest["car_detected"] = car_detected
            latest["guidance"] = guidance

        time.sleep(LOOP_DELAY_S)


def generate_mjpeg():
    while True:
        with state_lock:
            frame = latest["frame"]
        if frame is not None:
            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(LOOP_DELAY_S)


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        return jsonify({
            "offset_px": latest["offset_px"],
            "frame_width": latest["frame_width"],
            "distance_cm": latest["distance_cm"],
            "car_detected": latest["car_detected"],
            "guidance": latest["guidance"],
        })


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Garage Parking Assistant</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: -apple-system, sans-serif; background:#111; color:#eee; margin:0; padding:16px; }
  h1 { font-size:18px; margin:0 0 14px; }
  .video-wrap { border-radius:10px; overflow:hidden; border:1px solid #333; margin-bottom:16px; }
  img { width:100%; display:block; }
  .stats { display:flex; gap:12px; margin-bottom:16px; }
  .card { flex:1; background:#1c1c1c; border-radius:10px; padding:14px; text-align:center; }
  .card .label { font-size:11px; color:#999; text-transform:uppercase; letter-spacing:0.5px; }
  .card .value { font-size:26px; font-weight:600; margin-top:4px; }
  #guidance.stop { color:#ff4d4d; }
  #guidance.center { color:#4dff88; }
  #guidance.move { color:#ffd24d; }
  .gauge-wrap { background:#1c1c1c; border-radius:10px; padding:16px; }
  svg { display:block; margin:0 auto; }
</style>
</head>
<body>
  <h1>Smart Garage Parking Assistant</h1>

  <div class="video-wrap">
    <img src="/video_feed">
  </div>

  <div class="stats">
    <div class="card">
      <div class="label">Guidance</div>
      <div class="value" id="guidance">--</div>
    </div>
    <div class="card">
      <div class="label">Distance</div>
      <div class="value" id="distance">-- cm</div>
    </div>
  </div>

  <div class="gauge-wrap">
    <svg id="gauge" width="280" height="80" viewBox="0 0 280 80">
      <line x1="140" y1="10" x2="140" y2="60" stroke="#444" stroke-width="2" stroke-dasharray="4,4"/>
      <line x1="20" y1="35" x2="260" y2="35" stroke="#444" stroke-width="1"/>
      <circle id="dot" cx="140" cy="35" r="10" fill="#4dff88"/>
    </svg>
  </div>

<script>
async function poll() {
  try {
    const res = await fetch('/status');
    const d = await res.json();

    const g = document.getElementById('guidance');
    g.textContent = d.guidance;
    g.className = d.guidance === 'STOP' ? 'stop' : d.guidance === 'CENTER' ? 'center' : 'move';

    document.getElementById('distance').textContent =
      d.distance_cm !== null ? d.distance_cm.toFixed(0) + ' cm' : '-- cm';

    const dot = document.getElementById('dot');
    if (d.car_detected) {
      const scale = 240 / d.frame_width;
      let x = 140 + d.offset_px * scale;
      x = Math.max(30, Math.min(250, x));
      dot.setAttribute('cx', x);
      dot.setAttribute('fill', d.guidance === 'STOP' ? '#ff4d4d' : '#4dff88');
    } else {
      dot.setAttribute('fill', '#555');
    }
  } catch (e) {
    console.error(e);
  }
  setTimeout(poll, 200);
}
poll();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    t = threading.Thread(target=sensor_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
