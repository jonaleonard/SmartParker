import cv2
import numpy as np
import onnxruntime as ort

from calibration_config import load_roi_right_frac


class AlignmentCamera:
    """
    Uses an ONNX-exported YOLOv8 nano model to detect a car in frame and
    reports how far left/right it is from center. Works with any car (no
    markers needed on vehicle).
    """

    COCO_CAR_CLASS = 2  # "car" class ID in the COCO dataset YOLO is trained on

    def __init__(self, camera_index=0, width=1280, height=720, confidence=0.4,
                 imgsz=288, center_offset_px=0, roi_right_frac=None):
        """
        imgsz: retained for compatibility with existing callers. The exported
            ONNX model's input shape is used for inference (normally 640x640).
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
        """
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        capture_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.capture_width = capture_width
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.last_full_frame = None
        self.set_roi_right_frac(load_roi_right_frac() if roi_right_frac is None else roi_right_frac)

        self.model = ort.InferenceSession(
            "yolov8n.onnx", providers=["CPUExecutionProvider"]
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

    def set_roi_right_frac(self, roi_right_frac):
        """Set the right crop boundary as a fraction of the captured frame."""
        value = float(roi_right_frac)
        if not 0 < value <= 1:
            raise ValueError("roi_right_frac must be greater than 0 and at most 1")
        self.roi_right_frac = value
        self.frame_width = max(1, int(self.capture_width * value))

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
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.input_width - resized_width) / 2
        pad_y = (self.input_height - resized_height) / 2
        left, top = round(pad_x - 0.1), round(pad_y - 0.1)
        right, bottom = round(pad_x + 0.1), round(pad_y + 0.1)
        image = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[None]
        return image, scale, left, top

    def _car_boxes_from_output(self, output):
        """Extract car boxes and scores from raw or NMS-fused YOLOv8 output."""
        detections = np.squeeze(output)
        if detections.ndim != 2:
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

        # Standard Ultralytics YOLOv8 export: [1, 84, 8400] -> [8400, 84].
        # Values are xywh followed by the 80 per-class confidences.
        if detections.shape[0] in (84, 85) and detections.shape[1] > detections.shape[0]:
            detections = detections.T
        if detections.shape[1] >= 84:
            scores = detections[:, 4 + self.COCO_CAR_CLASS]
            selected = detections[scores >= self.confidence]
            scores = scores[scores >= self.confidence]
            xywh = selected[:, :4]
            boxes = np.column_stack((
                xywh[:, 0] - xywh[:, 2] / 2,
                xywh[:, 1] - xywh[:, 3] / 2,
                xywh[:, 0] + xywh[:, 2] / 2,
                xywh[:, 1] + xywh[:, 3] / 2,
            ))
            return boxes, scores

        # NMS-fused exports commonly use [batch, detections, 6] as
        # (x1, y1, x2, y2, confidence, class_id).
        if detections.shape[1] == 6:
            car = detections[(detections[:, 5].astype(int) == self.COCO_CAR_CLASS)
                             & (detections[:, 4] >= self.confidence)]
            return car[:, :4], car[:, 4]
        if detections.shape[0] == 6:
            return self._car_boxes_from_output(detections.T[None, ...])

        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

    def read_frame(self):
        """Grabs one raw frame from the camera and crops it to the watched ROI. Returns (ret, frame)."""
        ret, frame = self.cap.read()
        if not ret:
            return ret, frame
        self.last_full_frame = frame
        return ret, frame[:, :self.frame_width]

    def detect(self, frame):
        """
        Runs detection on an already-captured frame.
        Returns (offset_px, box_height_px, detected):
        - offset_px: negative = car is left of center, positive = right of center
        - box_height_px: height of detected car's bounding box (proxy for closeness,
          NOT a substitute for the ultrasonic sensor's actual distance reading)
        - detected: True if a car was found this frame
        """
        image, scale, pad_x, pad_y = self._preprocess(frame)
        outputs = self.model.run(None, {self.input_name: image})
        boxes, scores = self._car_boxes_from_output(outputs[0])
        if len(boxes) == 0:
            return 0, 0, False

        # Raw YOLOv8 ONNX outputs do not include NMS. Running it here is also
        # harmless for NMS-fused outputs and keeps behavior consistent.
        boxes = boxes[self._nms(boxes, scores)]

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

        return offset_px, box_height_px, True

    def get_alignment(self):
        """Convenience wrapper: reads a frame and detects in one call."""
        ret, frame = self.read_frame()
        if not ret:
            return 0, 0, False
        return self.detect(frame)

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    print("Starting alignment test, Ctrl+C to stop...")
    cam = AlignmentCamera()
    print(f"Watching {cam.frame_width}x{cam.frame_height} (roi_right_frac={cam.roi_right_frac}), imgsz={cam.imgsz}")

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
                      f"{offset + cam.center_offset_px:.1f}px)")
            else:
                print("No car detected")
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
