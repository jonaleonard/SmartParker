import cv2
from ultralytics import YOLO


class AlignmentCamera:
    """
    Uses YOLO nano to detect a car in frame and reports how far left/right
    it is from center. Works with any car (no markers needed on vehicle).
    """

    COCO_CAR_CLASS = 2  # "car" class ID in the COCO dataset YOLO is trained on

    def __init__(self, camera_index=0, width=1280, height=720, confidence=0.4,
                 imgsz=288, center_offset_px=0, roi_right_frac=1.0):
        """
        imgsz: side length (px) the frame is downscaled to before YOLO inference.
            Lower = faster but less accurate on small/far objects.
        center_offset_px: calibration constant that shifts what counts as "center"
            within the region actually being watched (see roi_right_frac below).
            Positive shifts the effective center right, negative shifts it left.
            See the __main__ block below for how to measure this.
        roi_right_frac: fraction (0-1] of the frame's width, measured from the
            LEFT edge, that is actually captured and analyzed — everything to
            the right of this line is cropped away before it ever reaches YOLO.
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
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.roi_right_frac = roi_right_frac
        self.frame_width = int(capture_width * roi_right_frac)

        # Auto-downloads yolov8n.pt (~6MB) on first run
        self.model = YOLO("yolov8n.pt")
        self.confidence = confidence
        self.imgsz = imgsz
        self.center_offset_px = center_offset_px

    def read_frame(self):
        """Grabs one raw frame from the camera and crops it to the watched ROI. Returns (ret, frame)."""
        ret, frame = self.cap.read()
        if not ret:
            return ret, frame
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
        results = self.model(
            frame,
            classes=[self.COCO_CAR_CLASS],
            conf=self.confidence,
            imgsz=self.imgsz,
            verbose=False,
        )

        boxes = results[0].boxes
        if len(boxes) == 0:
            return 0, 0, False

        # If multiple cars detected, use the largest box (closest/most prominent)
        areas = [(b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]) for b in boxes]
        best_idx = areas.index(max(areas))
        box = boxes[best_idx]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_center_x = (x1 + x2) / 2
        box_height_px = y2 - y1

        frame_center_x = self.frame_width / 2
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
    # Calibration, two steps:
    # 1. ROI: import ROI_RIGHT_FRAC from main.py/app.py below, or override it here,
    #    until roi_snapshot.jpg (saved each run) crops right where the left spot ends
    #    and the neighboring spot begins.
    # 2. Centering: with the car parked exactly where it should be, read the offset
    #    this prints and set CENTER_OFFSET_PX in main.py / app.py to that number so
    #    "centered in the spot" reads as 0.
    ROI_RIGHT_FRAC = 0.55  # keep in sync with main.py / app.py

    print("Starting alignment test, Ctrl+C to stop...")
    cam = AlignmentCamera(roi_right_frac=ROI_RIGHT_FRAC)
    print(f"Watching {cam.frame_width}x{cam.frame_height} (roi_right_frac={ROI_RIGHT_FRAC}), imgsz={cam.imgsz}")

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
