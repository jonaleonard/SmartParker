from time import sleep

from camera import AlignmentCamera
from distance_sensor import UltrasonicDistance
from display import ParkingDisplay
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
# spot) is cropped away before detection ever sees it. Tune by running
# `python3 camera.py`, which saves roi_snapshot.jpg so you can check the crop
# line lands where the left spot ends.
ROI_RIGHT_FRAC = 0.55

# Calibration offset (px, ROI-frame units) so "centered in the spot" reads
# as 0 even if the camera itself is mounted off-axis. Run `python3 camera.py`
# with the car parked exactly where it should be and set this to the offset
# it reports.
CENTER_OFFSET_PX = 0


def main():
    print("Starting Smart Garage Parking Assistant...")

    cam = AlignmentCamera(imgsz=DETECTION_IMGSZ, center_offset_px=CENTER_OFFSET_PX,
                           roi_right_frac=ROI_RIGHT_FRAC)
    ultrasonic = UltrasonicDistance()
    display = ParkingDisplay(stop_distance_cm=STOP_DISTANCE_CM)
    fusion = ParkingFusion(
        stop_distance_cm=STOP_DISTANCE_CM,
        center_tolerance_px=CENTER_TOLERANCE_PX,
    )

    print(f"Camera resolution: {cam.frame_width}x{cam.frame_height}")
    print("Running main loop. Ctrl+C to stop.")

    try:
        while True:
            offset_px, box_height_px, car_detected = cam.get_alignment()
            distance_cm = ultrasonic.get_distance_cm()

            guidance = fusion.get_guidance(offset_px, car_detected, distance_cm)

            display.show(
                offset_px=offset_px,
                frame_width=cam.frame_width,
                distance_cm=distance_cm,
                car_detected=car_detected,
            )

            print(f"{guidance:12} | dist: {distance_cm:6.1f} cm | offset: {offset_px:7.1f}px | car: {car_detected}")

            sleep(LOOP_DELAY_S)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cam.release()


if __name__ == "__main__":
    main()
