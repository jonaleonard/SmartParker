"""
Web dashboard for the Smart Garage Parking Assistant.
Serves a live webcam feed with detection boxes drawn on it, plus a
real-time alignment/distance gauge — viewable from any browser on the
same WiFi network (phone, Mac, etc). Also drives the OLED panel.

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
from display import ParkingDisplay
from fusion import ParkingFusion

STOP_DISTANCE_CM = 20
CENTER_TOLERANCE_PX = 40

# YOLO inference size in px. The exported ONNX model's own fixed input shape
# is what actually governs inference cost; this is kept for compatibility.
DETECTION_IMGSZ = 288

# Streaming is deliberately decoupled from detection. Inference costs ~550ms
# per frame on a Pi 4, so anything that waits on it can only ever show ~2fps
# of very stale video. The encoder instead publishes the newest camera frame
# at STREAM_FPS with the most recent detection drawn over it, which is what
# makes the feed look live.
STREAM_FPS = 15
STREAM_WIDTH = 640   # downscaled from the capture width; keeps latency and bandwidth low
JPEG_QUALITY = 60    # visibly fine for a demo, roughly a third the bytes of the default 95
CALIBRATION_STREAM_FPS = 6  # only viewed while someone is on the calibration page

DISPLAY_FPS = 5      # OLED refresh; the panel itself cannot usefully go faster
SENSOR_FPS = 10      # ultrasonic poll rate

JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

app = Flask(__name__)

cam = AlignmentCamera(
    imgsz=DETECTION_IMGSZ,
    center_offset_px=load_center_offset_px(),
    roi_right_frac=load_roi_right_frac(),
    roi_left_frac=load_roi_left_frac(),
)
ultrasonic = UltrasonicDistance()
fusion = ParkingFusion(stop_distance_cm=STOP_DISTANCE_CM, center_tolerance_px=CENTER_TOLERANCE_PX)

# The OLED is optional: a missing panel or a missing luma.oled install should
# degrade to a working web dashboard rather than stopping the whole app.
try:
    display = ParkingDisplay(stop_distance_cm=STOP_DISTANCE_CM)
    display_error = None
except Exception as error:  # noqa: BLE001 - any I2C/import failure is non-fatal here
    display = None
    display_error = str(error)
    print(f"WARNING: OLED display unavailable ({error}). Web dashboard will still run.")

# shared state, updated by the background loops, read by Flask routes
state_lock = threading.Lock()
latest = {
    "offset_px": 0,
    "frame_width": cam.frame_width,
    "distance_cm": None,
    "car_detected": False,
    "guidance": "STARTING",
    "box": None,
    "stream_fps": 0.0,
}


class FrameBroker:
    """
    Holds the most recently encoded JPEG and wakes streaming clients when a
    new one arrives. Encoding happens once no matter how many browsers are
    watching, and a client that cannot keep up simply skips ahead to the
    newest frame instead of accumulating a backlog.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._jpeg = None
        self._sequence = 0
        self._viewers = 0

    @property
    def viewers(self):
        with self._condition:
            return self._viewers

    def publish(self, jpeg):
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def wait_for_viewer(self, timeout=0.5):
        """Block while nobody is watching so idle streams cost no CPU."""
        with self._condition:
            return self._condition.wait_for(lambda: self._viewers > 0, timeout=timeout)

    def stream(self):
        with self._condition:
            self._viewers += 1
            self._condition.notify_all()
            last_sequence = self._sequence
        try:
            while True:
                with self._condition:
                    got_frame = self._condition.wait_for(
                        lambda: self._sequence != last_sequence, timeout=5.0)
                    if not got_frame:
                        continue
                    last_sequence = self._sequence
                    jpeg = self._jpeg
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        finally:
            with self._condition:
                self._viewers -= 1


dashboard_broker = FrameBroker()
calibration_broker = FrameBroker()


def detection_loop():
    """Runs YOLO as fast as it can manage, independent of the video stream."""
    while True:
        ret, roi_frame = cam.read_frame()
        if not ret:
            time.sleep(0.05)
            continue

        offset_px, box_height_px, car_detected = cam.detect(roi_frame)
        with state_lock:
            distance_cm = latest["distance_cm"]
        guidance = fusion.get_guidance(offset_px, car_detected, distance_cm)

        with state_lock:
            latest["offset_px"] = offset_px
            latest["frame_width"] = cam.frame_width
            latest["car_detected"] = car_detected
            latest["guidance"] = guidance
            latest["box"] = cam.last_box if car_detected else None


def sensor_loop():
    """Polls the ultrasonic sensor on its own schedule; it is far cheaper than vision."""
    interval = 1.0 / SENSOR_FPS
    while True:
        try:
            distance_cm = ultrasonic.get_distance_cm()
        except Exception:  # noqa: BLE001 - a bad reading should not kill the loop
            distance_cm = None
        with state_lock:
            latest["distance_cm"] = distance_cm
        time.sleep(interval)


def display_loop():
    """Drives the OLED from the same shared state the web dashboard reads."""
    if display is None:
        return
    interval = 1.0 / DISPLAY_FPS
    while True:
        with state_lock:
            offset_px = latest["offset_px"]
            frame_width = latest["frame_width"]
            distance_cm = latest["distance_cm"]
            car_detected = latest["car_detected"]
            guidance = latest["guidance"]
        try:
            display.show(offset_px=offset_px, frame_width=frame_width,
                         distance_cm=distance_cm, car_detected=car_detected,
                         guidance=guidance)
        except Exception as error:  # noqa: BLE001 - keep the app alive if I2C glitches
            print(f"WARNING: OLED update failed: {error}")
        time.sleep(interval)


def annotate(frame, scale):
    """Draw guidance overlays onto an already-downscaled ROI frame."""
    with state_lock:
        offset_px = latest["offset_px"]
        car_detected = latest["car_detected"]
        guidance = latest["guidance"]
        distance_cm = latest["distance_cm"]
        box = latest["box"]

    height, width = frame.shape[:2]

    # offset_px is relative to the calibrated garage center, not the camera's
    # raw geometric center, so the reference line has to shift by the same
    # calibration offset to match.
    garage_center_x = int((cam.frame_width / 2 + cam.center_offset_px) * scale)

    if car_detected:
        if box is not None:
            x1, y1, x2, y2 = (int(value * scale) for value in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        # The car marker is drawn as a middle band only, and the target line
        # over the top of it, so a perfectly centered car still shows both.
        marker_x = int(garage_center_x + offset_px * scale)
        cv2.line(frame, (marker_x, int(height * 0.35)), (marker_x, int(height * 0.65)),
                 (0, 0, 255), 3)

    cv2.line(frame, (garage_center_x, 0), (garage_center_x, height), (0, 255, 0), 2)

    color = (0, 0, 255) if guidance == "STOP" else (0, 255, 0) if guidance == "CENTER" else (0, 215, 255)
    cv2.putText(frame, guidance, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    distance_text = f"{distance_cm:.0f} cm" if distance_cm is not None else "-- cm"
    cv2.putText(frame, distance_text, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    stats = f"cam {cam.capture_fps_measured:.0f}fps  det {cam.detection_fps:.1f}fps"
    cv2.putText(frame, stats, (width - 250, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
    return frame


def encode_loop():
    """
    Publishes the newest camera frame at STREAM_FPS with the latest detection
    drawn on it. Never waits on inference, so the video stays live.
    """
    interval = 1.0 / STREAM_FPS
    last_sequence = None
    next_deadline = time.monotonic()
    fps_marker = time.monotonic()
    frames = 0

    while True:
        if not dashboard_broker.wait_for_viewer():
            continue

        frame, last_sequence = cam.latest_full_frame(since_seq=last_sequence, timeout=1.0)
        if frame is None:
            continue

        roi = cam.crop_to_roi(frame)
        scale = STREAM_WIDTH / roi.shape[1]
        if scale < 1:
            small = cv2.resize(roi, (STREAM_WIDTH, int(roi.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small, scale = roi.copy(), 1.0

        ok, jpeg = cv2.imencode(".jpg", annotate(small, scale), JPEG_PARAMS)
        if ok:
            dashboard_broker.publish(jpeg.tobytes())
            frames += 1

        now = time.monotonic()
        if now - fps_marker >= 1.0:
            with state_lock:
                latest["stream_fps"] = frames / (now - fps_marker)
            frames = 0
            fps_marker = now

        # Pace to STREAM_FPS without drifting, and without sleeping away time
        # already spent encoding.
        next_deadline = max(next_deadline + interval, now - interval)
        sleep_for = next_deadline - now
        if sleep_for > 0:
            time.sleep(sleep_for)


def calibration_encode_loop():
    """Streams the uncropped frame, only while the calibration page is open."""
    interval = 1.0 / CALIBRATION_STREAM_FPS
    last_sequence = None
    while True:
        if not calibration_broker.wait_for_viewer():
            continue

        frame, last_sequence = cam.latest_full_frame(since_seq=last_sequence, timeout=1.0)
        if frame is None:
            continue

        scale = STREAM_WIDTH / frame.shape[1]
        if scale < 1:
            frame = cv2.resize(frame, (STREAM_WIDTH, int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", frame, JPEG_PARAMS)
        if ok:
            calibration_broker.publish(jpeg.tobytes())
        time.sleep(interval)


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(dashboard_broker.stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/calibration_video_feed")
def calibration_video_feed():
    """Full camera image so the ROI boundary can be positioned visually."""
    return Response(calibration_broker.stream(),
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
            "stream_fps": round(latest["stream_fps"], 1),
            "capture_fps": round(cam.capture_fps_measured, 1),
            "detection_fps": round(cam.detection_fps, 2),
            "display_ok": display is not None,
            "display_error": display_error,
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
  #perf { color:#777; font-size:12px; margin-top:12px; text-align:center; }
  #perf .warn { color:#ff8a4d; }
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

  <div id="perf"></div>

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

    document.getElementById('perf').innerHTML =
      'video ' + d.stream_fps + ' fps &middot; camera ' + d.capture_fps + ' fps &middot; detection '
      + d.detection_fps + ' fps'
      + (d.display_ok ? '' : ' &middot; <span class="warn">OLED offline</span>');
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
    for loop in (sensor_loop, detection_loop, encode_loop, calibration_encode_loop, display_loop):
        threading.Thread(target=loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
