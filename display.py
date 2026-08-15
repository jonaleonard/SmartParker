from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from PIL import Image, ImageDraw


class ParkingDisplay:
    """
    Drives the 1.3" SH1106 OLED (128x64). Shows a center-line alignment
    graphic plus distance/STOP text.
    """

    WIDTH = 128
    HEIGHT = 64

    def __init__(self, i2c_port=1, i2c_address=0x3C, stop_distance_cm=20):
        serial = i2c(port=i2c_port, address=i2c_address)
        self.device = sh1106(serial)
        self.stop_distance_cm = stop_distance_cm

    def show(self, offset_px, frame_width, distance_cm, car_detected):
        """
        offset_px: pixel offset from camera (negative=left, positive=right)
        frame_width: width of the camera frame, used to scale offset to the OLED
        distance_cm: current ultrasonic distance reading
        car_detected: whether the camera currently sees a car
        """
        img = Image.new("1", (self.WIDTH, self.HEIGHT))
        draw = ImageDraw.Draw(img)

        if distance_cm is not None and distance_cm <= self.stop_distance_cm:
            self._draw_stop(draw)
        else:
            self._draw_guidance(draw, offset_px, frame_width, distance_cm, car_detected)

        self.device.display(img)

    def _draw_stop(self, draw):
        draw.rectangle([0, 0, self.WIDTH, self.HEIGHT], outline=255, fill=0)
        draw.text((30, 22), "STOP", fill=255)

    def _draw_guidance(self, draw, offset_px, frame_width, distance_cm, car_detected):
        center_x = self.WIDTH // 2

        # center reference line
        draw.line([(center_x, 0), (center_x, 40)], fill=255)
        draw.line([(4, 20), (124, 20)], fill=255)

        if car_detected and frame_width:
            # scale camera-frame offset down to OLED width, clamp to stay on screen
            scale = self.WIDTH / frame_width
            marker_x = center_x + (offset_px * scale)
            marker_x = max(6, min(self.WIDTH - 6, marker_x))
            draw.ellipse([marker_x - 4, 16, marker_x + 4, 24], fill=255)
        else:
            draw.text((10, 12), "No car detected", fill=255)

        dist_text = f"{distance_cm:.0f} cm" if distance_cm is not None else "-- cm"
        draw.text((10, 48), dist_text, fill=255)


if __name__ == "__main__":
    from time import sleep

    print("Testing display, Ctrl+C to stop...")
    disp = ParkingDisplay()
    try:
        # simulate a car drifting left to right while approaching
        offset = -300
        distance = 150
        while True:
            disp.show(offset_px=offset, frame_width=1280, distance_cm=distance, car_detected=True)
            offset += 20
            distance -= 3
            if distance < 10:
                distance = 150
                offset = -300
            sleep(0.2)
    except KeyboardInterrupt:
        pass
