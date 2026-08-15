from gpiozero import DistanceSensor


class UltrasonicDistance:
    """
    Wraps the HC-SR04 ultrasonic sensor. This is the primary, reliable
    source of distance-to-car — vision's box_height is only a rough proxy.
    """

    def __init__(self, echo_pin=24, trigger_pin=23, max_distance_m=4.0):
        self.sensor = DistanceSensor(
            echo=echo_pin,
            trigger=trigger_pin,
            max_distance=max_distance_m,
        )

    def get_distance_cm(self):
        """Returns distance to nearest object in centimeters."""
        return round(self.sensor.distance * 100, 1)


if __name__ == "__main__":
    from time import sleep

    print("Starting distance test, Ctrl+C to stop...")
    sensor = UltrasonicDistance()
    try:
        while True:
            print(f"Distance: {sensor.get_distance_cm()} cm")
            sleep(0.3)
    except KeyboardInterrupt:
        pass
