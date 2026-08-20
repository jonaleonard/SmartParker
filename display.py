from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(size):
    """Return a scalable font, falling back to PIL's tiny bitmap default."""
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


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
        self.device.show()  # a panel left in low-power sleep stays blank otherwise
        self.stop_distance_cm = stop_distance_cm
        self.big_font = _load_font(22)
        self.font = _load_font(13)
        self.small_font = _load_font(10)

    def _centered_text(self, draw, y, text, font):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text(((self.WIDTH - (right - left)) / 2 - left, y), text, font=font, fill=255)

    def show(self, offset_px, frame_width, distance_cm, car_detected, guidance=None):
        """
        offset_px: pixel offset from camera (negative=left, positive=right)
        frame_width: width of the camera frame, used to scale offset to the OLED
        distance_cm: current ultrasonic distance reading
        car_detected: whether the camera currently sees a car
        guidance: optional fused guidance string, shown verbatim when given
        """
        img = Image.new("1", (self.WIDTH, self.HEIGHT))
        draw = ImageDraw.Draw(img)

        stopping = distance_cm is not None and distance_cm <= self.stop_distance_cm
        if stopping or guidance == "STOP":
            self._draw_stop(draw, distance_cm)
        else:
            self._draw_guidance(draw, offset_px, frame_width, distance_cm,
                                car_detected, guidance)

        self.device.display(img)

    def _draw_stop(self, draw, distance_cm):
        draw.rectangle([0, 0, self.WIDTH - 1, self.HEIGHT - 1], outline=255, fill=0)
        self._centered_text(draw, 14, "STOP", self.big_font)
        distance_text = f"{distance_cm:.0f} cm" if distance_cm is not None else "-- cm"
        self._centered_text(draw, 44, distance_text, self.font)

    def _draw_guidance(self, draw, offset_px, frame_width, distance_cm,
                       car_detected, guidance):
        center_x = self.WIDTH // 2

        if car_detected and frame_width:
            # Alignment bar: a fixed center tick with a marker that slides to
            # show which way the car is off, scaled from camera pixels.
            draw.line([(center_x, 2), (center_x, 14)], fill=255)
            draw.line([(4, 22), (self.WIDTH - 5, 22)], fill=255)

            scale = self.WIDTH / frame_width
            marker_x = center_x + (offset_px * scale)
            marker_x = max(7, min(self.WIDTH - 7, marker_x))
            draw.ellipse([marker_x - 5, 17, marker_x + 5, 27], fill=255)

            label = guidance if guidance else ("CENTER" if abs(offset_px * scale) < 4 else
                                               "MOVE RIGHT" if offset_px > 0 else "MOVE LEFT")
            self._centered_text(draw, 32, label, self.font)
        else:
            self._centered_text(draw, 16, "NO CAR", self.big_font)

        distance_text = f"{distance_cm:.0f} cm" if distance_cm is not None else "-- cm"
        self._centered_text(draw, 50, distance_text, self.font)


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
    finally:
        disp.device.clear()
