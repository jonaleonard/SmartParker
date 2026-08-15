class ParkingFusion:
    """
    Combines camera alignment (left/right) with ultrasonic distance
    (how close) into a single guidance state. Ultrasonic is treated as
    the source of truth for distance/STOP; vision handles alignment.
    """

    def __init__(self, stop_distance_cm=20, center_tolerance_px=40):
        self.stop_distance_cm = stop_distance_cm
        self.center_tolerance_px = center_tolerance_px

    def get_guidance(self, offset_px, car_detected, distance_cm):
        if distance_cm is not None and distance_cm <= self.stop_distance_cm:
            return "STOP"

        if not car_detected:
            return "NO CAR"

        if abs(offset_px) <= self.center_tolerance_px:
            return "CENTER"
        elif offset_px > 0:
            return "MOVE RIGHT"
        else:
            return "MOVE LEFT"
