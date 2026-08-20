from time import sleep

from calibration_config import load_center_offset_px
from camera import AlignmentCamera
from distance_sensor import UltrasonicDistance
from display import ParkingDisplay
from fusion import ParkingFusion

STOP_DISTANCE_CM = 20
CENTER_TOLERANCE_PX = 40
LOOP_DELAY_S = 0.02  # small yield only; detection itself (~0.2-0.3s) sets the real pace

# YOLO inference size in px; lower = faster but less accurate on small/far
# objects. The calibrated ROI crop narrows the wide 16:9 frame before resize.
DETECTION_IMGSZ = 288

def main():
    print("Starting Smart Garage Parking Assistant...")

    cam = AlignmentCamera(imgsz=DETECTION_IMGSZ,
                           center_offset_px=load_center_offset_px())
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
                guidance=guidance,
            )

            print(f"{guidance:12} | dist: {distance_cm:6.1f} cm | offset: {offset_px:7.1f}px "
                  f"| car: {car_detected} | cam {cam.capture_fps_measured:.1f}fps "
                  f"| detect {cam.detection_fps:.2f}fps")

            sleep(LOOP_DELAY_S)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cam.release()
        display.device.clear()


if __name__ == "__main__":
    main()
