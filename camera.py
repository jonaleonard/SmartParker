import threading
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

from calibration_config import load_roi_left_frac, load_roi_right_frac


class AlignmentCamera:
    """
    Uses an ONNX-exported YOLOv8 nano model to detect a car in frame and
    reports how far left/right it is from center. Works with any car (no
    markers needed on vehicle).

    Capture runs on its own thread that always keeps only the newest frame,
    so a slow consumer (detection takes far longer than a frame interval)
    never falls behind on a backlog of stale buffered frames.
    """

    COCO_CAR_CLASS = 2  # "car" class ID in the COCO dataset YOLO is trained on
    LOCK_ON_WINDOW_FRAMES = 6
    LOCK_ON_HITS = 2
    RELEASE_AFTER_MISSES = 8

    def __init__(self, camera_index=0, width=1280, height=720, confidence=0.2,
                 imgsz=480, center_offset_px=0, roi_right_frac=None,
                 roi_left_frac=None, capture_fps=30, model_path="yolov8n.onnx",
                 num_threads=3, debug=False):
        """
        imgsz: retained for compatibility with existing callers. The exported
            ONNX model's input shape is used for inference.
        center_offset_px: calibration constant that shifts what counts as "center"
            within the region actually being watched (see roi_right_frac below).
            Positive shifts the effective center right, negative shifts it left.
            Set this with the web dashboard's /calibrate page.
        roi_right_frac: fraction (0-1] of the frame's width, measured from the
            LEFT edge, that is actually captured and analyzed — everything to
            the right of this line is cropped away before it ever reaches YOLO.
            When omitted, the value is loaded from config.json and defaults to
            1.0 (the complete frame) if no saved value is available.
            Use this when only one side of a shared space (e.g. one spot in a
            two-car garage) should ever be considered; it removes any chance
            of the other side's car being picked up, and as a side effect
            shrinks a wide 16:9 frame toward square, which YOLO's letterbox
            resize handles far more accurately than a very wide frame.
        roi_left_frac: fraction [0-1) of the frame's width, measured from the
            left edge. Everything to the left of this line is cropped away.
            Together with roi_right_frac, it defines the watched region.
        capture_fps: frame rate requested from the camera. MJPG is requested
            too; without it a USB webcam negotiates raw YUYV, which saturates
            the USB bus and caps 720p at about 5 fps.
        num_threads: ONNX inference threads. Left at 3 of the Pi's 4 cores so
            capture and the web server still get scheduled promptly.
        debug: print per-frame detection diagnostics. Off by default — at a
            few frames a second the writes alone measurably slow the loop.
        """
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        # Order matters: the pixel format has to be selected before the frame
        # size, or the driver picks a size valid only for the old format.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, capture_fps)
        # One-deep queue: combined with the reader thread below, a read always
        # returns something close to live rather than a queued older frame.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        capture_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.capture_width = capture_width
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.debug = debug

        self.last_full_frame = None
        self.set_roi_bounds(
            load_roi_left_frac() if roi_left_frac is None else roi_left_frac,
            load_roi_right_frac() if roi_right_frac is None else roi_right_frac,
        )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = ort.InferenceSession(
            model_path, sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        model_input = self.model.get_inputs()[0]
        self.input_name = model_input.name
        # yolov8n.onnx is ordinarily exported with [1, 3, 640, 640].  Read
        # this from the model so a differently exported fixed-size model works
        # without code changes.
        input_shape = model_input.shape
        self.input_height = self._static_dimension(input_shape[2], 640)
        self.input_width = self._static_dimension(input_shape[3], 640)
        self.confidence = confidence
        self.imgsz = imgsz
        self.center_offset_px = center_offset_px
        self._printed_output_debug = False
        self._detection_history = deque(maxlen=self.LOCK_ON_WINDOW_FRAMES)
        self._last_valid_detection = None
        self._locked_on = False
        self._consecutive_misses = 0

        # Box of the most recent raw detection, in ROI pixel coordinates, so
        # callers can draw it. None whenever nothing is currently locked on.
        self.last_box = None
        self.detection_fps = 0.0
        self.capture_fps_measured = 0.0

        self._frame_condition = threading.Condition()
        self._latest_frame = None
        self._frame_seq = 0
        self._running = True
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self._await_first_frame()

    def _reader_loop(self):
        """Continuously drain the camera so only the newest frame is kept."""
        interval_samples = deque(maxlen=30)
        previous = time.monotonic()
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                # A transient USB hiccup should not spin the CPU or kill the
                # thread; back off briefly and keep trying.
                time.sleep(0.05)
                continue

            now = time.monotonic()
            interval_samples.append(now - previous)
            previous = now

            with self._frame_condition:
                self._latest_frame = frame
                self._frame_seq += 1
                if interval_samples:
                    average = sum(interval_samples) / len(interval_samples)
                    self.capture_fps_measured = 1.0 / average if average > 0 else 0.0
                self._frame_condition.notify_all()

    def _await_first_frame(self, timeout=5.0):
        with self._frame_condition:
            self._frame_condition.wait_for(lambda: self._latest_frame is not None,
                                           timeout=timeout)

    def latest_full_frame(self, since_seq=None, timeout=1.0):
        """
        Return (frame, seq) for the newest captured frame.

        When since_seq is given, block until a frame newer than it arrives (or
        the timeout expires), so streaming consumers wake exactly once per new
        frame instead of polling and re-sending duplicates.
        """
        with self._frame_condition:
            if since_seq is not None:
                self._frame_condition.wait_for(lambda: self._frame_seq > since_seq,
                                               timeout=timeout)
            return self._latest_frame, self._frame_seq

    def crop_to_roi(self, frame):
        """Crop a full camera frame down to the calibrated watched region."""
        return frame[:, self._roi_slice]

    def _smoothed_detection(self, offset_px, box_height_px, detected):
        """Apply lock/release hysteresis while retaining the latest geometry."""
        self._detection_history.append(detected)
        if detected:
            self._last_valid_detection = (offset_px, box_height_px)
            self._consecutive_misses = 0
            if not self._locked_on:
                self._locked_on = sum(self._detection_history) >= self.LOCK_ON_HITS
        elif self._locked_on:
            self._consecutive_misses += 1
            if self._consecutive_misses >= self.RELEASE_AFTER_MISSES:
                self._locked_on = False

        if not self._locked_on:
            self.last_box = None
            return 0, 0, False
        if detected:
            return offset_px, box_height_px, True

        # A lock keeps the car present through intermittent inference misses.
        return (*self._last_valid_detection, True)

    def set_roi_right_frac(self, roi_right_frac):
        """Set the right crop boundary, preserving the current left boundary."""
        self.set_roi_bounds(getattr(self, "roi_left_frac", 0.0), roi_right_frac)

    def set_roi_left_frac(self, roi_left_frac):
        """Set the left crop boundary, preserving the current right boundary."""
        self.set_roi_bounds(roi_left_frac, getattr(self, "roi_right_frac", 1.0))

    def set_roi_bounds(self, roi_left_frac, roi_right_frac):
        """Set both ROI boundaries as fractions of the captured frame."""
        left_value = float(roi_left_frac)
        right_value = float(roi_right_frac)
        if not 0 <= left_value < 1:
            raise ValueError("roi_left_frac must be at least 0 and less than 1")
        if not 0 < right_value <= 1:
            raise ValueError("roi_right_frac must be greater than 0 and at most 1")

        left_px = int(self.capture_width * left_value)
        right_px = int(self.capture_width * right_value)
        if left_px >= right_px:
            raise ValueError("ROI left boundary must be to the left of the right boundary")

        self.roi_left_frac = left_value
        self.roi_right_frac = right_value
        self.roi_left_px = left_px
        self.roi_right_px = right_px
        self.frame_width = right_px - left_px
        # Single atomic handle for the crop, so a frame being cropped while
        # calibration is saved can never mix a new left edge with an old right.
        self._roi_slice = slice(left_px, right_px)

    @staticmethod
    def _static_dimension(value, fallback):
        """Return an ONNX input dimension, falling back for dynamic models."""
        return value if isinstance(value, int) and value > 0 else fallback

    @staticmethod
    def _nms(boxes, scores, iou_threshold=0.45):
        """Return indices of boxes kept by class-specific non-max suppression."""
        if len(boxes) == 0:
            return []

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size:
            current = order[0]
            keep.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            inter_x1 = np.maximum(x1[current], x1[remaining])
            inter_y1 = np.maximum(y1[current], y1[remaining])
            inter_x2 = np.minimum(x2[current], x2[remaining])
            inter_y2 = np.minimum(y2[current], y2[remaining])
            intersection = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
            union = areas[current] + areas[remaining] - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = remaining[iou <= iou_threshold]

        return keep

    def _preprocess(self, frame):
        """Letterbox an OpenCV BGR frame and convert it to YOLOv8 NCHW input."""
        height, width = frame.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        # INTER_AREA is both faster and cleaner than INTER_LINEAR when
        # shrinking, which is always the direction here.
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        pad_x = (self.input_width - resized_width) / 2
        pad_y = (self.input_height - resized_height) / 2
        left, top = round(pad_x - 0.1), round(pad_y - 0.1)
        right, bottom = round(pad_x + 0.1), round(pad_y + 0.1)
        image = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[None]
        return np.ascontiguousarray(image), scale, left, top

    def _car_boxes_from_output(self, output):
        """Extract car boxes, scores, and pre-NMS debug counts from YOLO output."""
        detections = np.squeeze(output)
        if detections.ndim != 2:
            return (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32),
                    0, 0, float("nan"))

        # Standard Ultralytics YOLOv8 export: [1, 84, 8400] -> [8400, 84].
        # Values are xywh followed by the 80 per-class confidences.
        if detections.shape[0] in (84, 85) and detections.shape[1] > detections.shape[0]:
            detections = detections.T
        if detections.shape[1] >= 84:
            # For [cx, cy, w, h, class0, ..., class79], score row 6 is the
            # COCO car score (4 + class 2).  Do not use an all-class argmax:
            # that would retain anchors whose strongest class is not a car.
            car_scores = detections[:, 4 + self.COCO_CAR_CLASS]
            highest_raw_score = float(np.max(car_scores)) if car_scores.size else float("nan")
            keep = car_scores >= self.confidence
            # The car-score threshold is the class filter for a raw YOLOv8
            # output, so this count must reflect only retained car anchors.
            class_count = int(np.count_nonzero(keep))
            selected = detections[keep]
            scores = car_scores[keep]
            xywh = selected[:, :4]
            boxes = np.column_stack((
                xywh[:, 0] - xywh[:, 2] / 2,
                xywh[:, 1] - xywh[:, 3] / 2,
                xywh[:, 0] + xywh[:, 2] / 2,
                xywh[:, 1] + xywh[:, 3] / 2,
            ))
            return boxes, scores, class_count, len(boxes), highest_raw_score

        # NMS-fused exports commonly use [batch, detections, 6] as
        # (x1, y1, x2, y2, confidence, class_id).
        if detections.shape[1] == 6:
            highest_raw_score = float(np.max(detections[:, 4])) if len(detections) else float("nan")
            car = detections[detections[:, 5].astype(int) == self.COCO_CAR_CLASS]
            selected = car[car[:, 4] >= self.confidence]
            return selected[:, :4], selected[:, 4], len(car), len(selected), highest_raw_score
        if detections.shape[0] == 6:
            return self._car_boxes_from_output(detections.T[None, ...])

        return (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32),
                0, 0, float("nan"))

    def read_frame(self):
        """Returns (ret, frame) for the newest captured frame, cropped to the watched ROI."""
        frame, _ = self.latest_full_frame()
        if frame is None:
            return False, None
        self.last_full_frame = frame
        return True, self.crop_to_roi(frame)

    def detect(self, frame):
        """
        Runs detection on an already-captured frame.
        Returns (offset_px, box_height_px, detected):
        - offset_px: negative = car is left of center, positive = right of center
        - box_height_px: height of detected car's bounding box (proxy for closeness,
          NOT a substitute for the ultrasonic sensor's actual distance reading)
        - detected: True while the car is locked on, including brief raw misses
        """
        started = time.monotonic()
        image, scale, pad_x, pad_y = self._preprocess(frame)
        outputs = self.model.run(None, {self.input_name: image})
        boxes, scores, class_count, confidence_count, highest_raw_score = self._car_boxes_from_output(outputs[0])

        elapsed = time.monotonic() - started
        self.detection_fps = 1.0 / elapsed if elapsed > 0 else 0.0

        if self.debug:
            if not self._printed_output_debug:
                print(f"DEBUG raw ONNX output shape: {outputs[0].shape}")
                self._printed_output_debug = True
            print(f"DEBUG highest raw confidence: {highest_raw_score:.6f}")
            print(f"DEBUG boxes after car class filter (class {self.COCO_CAR_CLASS}): {class_count}")
            print(f"DEBUG boxes after confidence threshold ({self.confidence:.2f}): {confidence_count}")
            print(f"DEBUG inference: {elapsed * 1000:.0f}ms")

        if len(boxes) == 0:
            if self.debug:
                print("DEBUG boxes after NMS: 0")
            return self._smoothed_detection(0, 0, False)

        # Raw YOLOv8 ONNX outputs do not include NMS. Running it here is also
        # harmless for NMS-fused outputs and keeps behavior consistent.
        boxes = boxes[self._nms(boxes, scores)]
        if self.debug:
            print(f"DEBUG boxes after NMS: {len(boxes)}")

        # Convert letterboxed model coordinates back to the captured ROI.
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        frame_height, frame_width = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_height)

        # If multiple cars detected, use the largest box (closest/most prominent)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box = boxes[np.argmax(areas)]

        x1, y1, x2, y2 = box.tolist()
        box_center_x = (x1 + x2) / 2
        box_height_px = y2 - y1

        frame_center_x = frame_width / 2
        offset_px = box_center_x - frame_center_x - self.center_offset_px

        self.last_box = (int(x1), int(y1), int(x2), int(y2))
        return self._smoothed_detection(offset_px, box_height_px, True)

    def get_alignment(self):
        """Convenience wrapper: reads a frame and detects in one call."""
        ret, frame = self.read_frame()
        if not ret:
            return 0, 0, False
        return self.detect(frame)

    def release(self):
        self._running = False
        self._reader.join(timeout=2.0)
        self.cap.release()


if __name__ == "__main__":
    print("Starting alignment test, Ctrl+C to stop...")
    cam = AlignmentCamera(roi_right_frac=None, roi_left_frac=None, confidence=0.2,
                          imgsz=480, debug=True)
    print(f"Watching {cam.frame_width}x{cam.frame_height} "
          f"(roi_left_frac={cam.roi_left_frac}, roi_right_frac={cam.roi_right_frac}), "
          f"model input {cam.input_width}x{cam.input_height}")

    ret, frame = cam.read_frame()
    if ret:
        cv2.imwrite("roi_snapshot.jpg", frame)
        print("Saved roi_snapshot.jpg -- check that it's cropped where you want.")

    try:
        while True:
            offset, box_height, found = cam.get_alignment()
            if found:
                direction = "LEFT" if offset > 0 else "RIGHT" if offset < 0 else "CENTER"
                print(f"Offset: {offset:.1f}px -> move {direction} | box height: {box_height:.0f}px "
                      f"(raw, uncalibrated offset from ROI's own center: "
                      f"{offset + cam.center_offset_px:.1f}px) | "
                      f"capture {cam.capture_fps_measured:.1f}fps, detect {cam.detection_fps:.1f}fps")
            else:
                print("No car detected")
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
