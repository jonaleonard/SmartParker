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
from flask import Flask, Response, jsonify, render_template_string, request

from calibration_config import (load_center_offset_px, load_roi_left_frac,
                                load_roi_right_frac, save_calibration)
from camera import AlignmentCamera
from distance_sensor import UltrasonicDistance
from fusion import ParkingFusion

STOP_DISTANCE_CM = 20
CENTER_TOLERANCE_PX = 40
LOOP_DELAY_S = 0.02  # small yield only; detection itself (~0.2-0.3s) sets the real pace

# YOLO inference size in px; lower = faster but less accurate on small/far.
# The saved ROI crop narrows the wide camera image before it is resized.
DETECTION_IMGSZ = 288

app = Flask(__name__)

cam = AlignmentCamera(
    imgsz=DETECTION_IMGSZ,
    center_offset_px=load_center_offset_px(),
    roi_right_frac=load_roi_right_frac(),
    roi_left_frac=load_roi_left_frac(),
)
ultrasonic = UltrasonicDistance()
fusion = ParkingFusion(stop_distance_cm=STOP_DISTANCE_CM, center_tolerance_px=CENTER_TOLERANCE_PX)

# shared state, updated by the background loop, read by Flask routes
state_lock = threading.Lock()
latest = {
    "frame": None,
    "calibration_frame": None,
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
            latest["calibration_frame"] = cam.last_full_frame
            latest["offset_px"] = offset_px
            latest["frame_width"] = cam.frame_width
            latest["distance_cm"] = distance_cm
            latest["car_detected"] = car_detected
            latest["guidance"] = guidance

        time.sleep(LOOP_DELAY_S)


def generate_mjpeg(frame_key="frame"):
    while True:
        with state_lock:
            frame = latest[frame_key]
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


@app.route("/calibration_video_feed")
def calibration_video_feed():
    """Full camera image so the ROI boundary can be positioned visually."""
    return Response(generate_mjpeg("calibration_frame"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/calibrate")
def calibrate():
    return render_template_string(CALIBRATION_HTML)


@app.route("/calibration", methods=["GET", "POST"])
def calibration():
    if request.method == "GET":
        return jsonify({
            "center_offset_px": cam.center_offset_px,
            "frame_width": cam.frame_width,
            "capture_width": cam.capture_width,
            "roi_left_frac": cam.roi_left_frac,
            "roi_right_frac": cam.roi_right_frac,
        })

    data = request.get_json(silent=True) or {}
    try:
        center_offset_px = int(data["center_offset_px"])
        roi_left_frac = float(data["roi_left_frac"])
        roi_right_frac = float(data["roi_right_frac"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "center_offset_px must be an integer and ROI boundaries must be numbers"}), 400

    if not 0 <= roi_left_frac < 1:
        return jsonify({"error": "roi_left_frac must be at least 0 and less than 1"}), 400
    if not 0 < roi_right_frac <= 1:
        return jsonify({"error": "roi_right_frac must be greater than 0 and at most 1"}), 400
    roi_left_px = int(cam.capture_width * roi_left_frac)
    roi_right_px = int(cam.capture_width * roi_right_frac)
    if roi_left_px >= roi_right_px:
        return jsonify({"error": "ROI left boundary must be to the left of the right boundary"}), 400

    # Keep the selected center inside the newly selected ROI frame.
    roi_width = roi_right_px - roi_left_px
    half_width = roi_width / 2
    if not -half_width <= center_offset_px <= half_width:
        return jsonify({"error": "center_offset_px is outside the video frame"}), 400

    center_offset_px, roi_right_frac, roi_left_frac = save_calibration(
        center_offset_px, roi_right_frac, roi_left_frac)
    cam.center_offset_px = center_offset_px
    cam.set_roi_bounds(roi_left_frac, roi_right_frac)
    return jsonify({"center_offset_px": center_offset_px,
                    "roi_left_frac": roi_left_frac,
                    "roi_right_frac": roi_right_frac,
                    "frame_width": cam.frame_width,
                    "capture_width": cam.capture_width})


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
  <p><a href="/calibrate">Calibrate parking-spot center</a></p>

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


CALIBRATION_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Calibrate Parking Center</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family:-apple-system,sans-serif; background:#111; color:#eee; margin:0; padding:16px; }
  h1 { font-size:20px; margin:0 0 8px; }
  p { color:#bbb; line-height:1.4; }
  #video-wrap { position:relative; border:1px solid #444; border-radius:10px; overflow:hidden; touch-action:none; }
  #video { width:100%; display:block; }
  .calibration-line { position:absolute; top:0; bottom:0; width:4px; transform:translateX(-50%); cursor:ew-resize; box-shadow:0 0 4px #000; }
  .calibration-line::before { content:''; position:absolute; left:-10px; right:-10px; top:0; bottom:0; }
  #center-line { background:#ffcf33; }
  #roi-line { background:#43b9ff; }
  #roi-left-line { background:#c86bff; }
  .actions { display:flex; align-items:center; gap:12px; margin-top:16px; }
  button { background:#ffcf33; border:0; border-radius:7px; padding:10px 16px; font-size:16px; font-weight:600; color:#111; }
  #message { color:#bbb; }
  a { color:#8ac7ff; }
</style>
</head>
<body>
  <h1>Calibrate parking spot</h1>
  <p>Drag the yellow line to the parking-spot center, the purple line to its left edge, and the blue line to its right edge. The region between the boundary lines is used for parking guidance.</p>
  <div id="video-wrap">
    <img id="video" src="/calibration_video_feed" alt="Live garage camera feed">
    <div id="center-line" class="calibration-line" title="Drag to set parking-spot center"></div>
    <div id="roi-left-line" class="calibration-line" title="Drag to set parking-spot left edge"></div>
    <div id="roi-line" class="calibration-line" title="Drag to set parking-spot right edge"></div>
  </div>
  <div class="actions">
    <button id="save">Save</button><span id="message"></span>
  </div>
  <p><a href="/">Back to dashboard</a></p>
<script>
let captureWidth = 0;
let centerOffset = 0;
let roiLeftFrac = 0;
let roiRightFrac = 1;
let draggingLine = null;
const videoWrap = document.getElementById('video-wrap');
const centerLine = document.getElementById('center-line');
const roiLeftLine = document.getElementById('roi-left-line');
const roiLine = document.getElementById('roi-line');
const message = document.getElementById('message');

function drawLines() {
  if (!captureWidth) return;
  const roiLeftPx = captureWidth * roiLeftFrac;
  const roiWidth = captureWidth * roiRightFrac - roiLeftPx;
  centerLine.style.left = ((roiLeftPx + roiWidth / 2 + centerOffset) / captureWidth * 100) + '%';
  roiLeftLine.style.left = (roiLeftFrac * 100) + '%';
  roiLine.style.left = (roiRightFrac * 100) + '%';
}

function pointerFraction(event) {
  const bounds = videoWrap.getBoundingClientRect();
  return Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
}

function setLineFromPointer(event) {
  const fraction = pointerFraction(event);
  const centerFraction = (captureWidth * roiLeftFrac
                          + (captureWidth * roiRightFrac - captureWidth * roiLeftFrac) / 2
                          + centerOffset) / captureWidth;
  if (draggingLine === 'roi') {
    roiRightFrac = Math.max(roiLeftFrac + 1 / captureWidth, centerFraction, fraction);
  } else if (draggingLine === 'roi-left') {
    roiLeftFrac = Math.min(roiRightFrac - 1 / captureWidth, centerFraction, fraction);
  } else {
    const centerPx = Math.max(captureWidth * roiLeftFrac,
                              Math.min(captureWidth * roiRightFrac, fraction * captureWidth));
    centerOffset = Math.round(centerPx
                              - (captureWidth * roiLeftFrac
                                 + (captureWidth * roiRightFrac - captureWidth * roiLeftFrac) / 2));
  }
  drawLines();
}

fetch('/calibration').then(response => response.json()).then(data => {
  captureWidth = data.capture_width;
  centerOffset = data.center_offset_px;
  roiLeftFrac = data.roi_left_frac;
  roiRightFrac = data.roi_right_frac;
  drawLines();
}).catch(() => { message.textContent = 'Unable to load current calibration.'; });

function startDragging(which, event) {
  if (!captureWidth) return;
  draggingLine = which;
  event.currentTarget.setPointerCapture(event.pointerId);
  setLineFromPointer(event);
}

for (const [which, line] of [['center', centerLine], ['roi-left', roiLeftLine], ['roi', roiLine]]) {
  line.addEventListener('pointerdown', event => startDragging(which, event));
  line.addEventListener('pointermove', event => { if (draggingLine === which) setLineFromPointer(event); });
  line.addEventListener('pointerup', () => { draggingLine = null; });
  line.addEventListener('pointercancel', () => { draggingLine = null; });
}

document.getElementById('save').addEventListener('click', async () => {
  message.textContent = 'Saving…';
  try {
    const response = await fetch('/calibration', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({center_offset_px: centerOffset, roi_left_frac: roiLeftFrac,
                            roi_right_frac: roiRightFrac})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Save failed');
    centerOffset = data.center_offset_px;
    roiLeftFrac = data.roi_left_frac;
    roiRightFrac = data.roi_right_frac;
    captureWidth = data.capture_width;
    drawLines();
    message.textContent = 'Saved.';
  } catch (error) {
    message.textContent = error.message;
  }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    t = threading.Thread(target=sensor_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
