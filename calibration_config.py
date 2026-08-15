"""Persistent calibration settings shared by the web and display apps."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


CONFIG_PATH = Path(__file__).with_name("config.json")


def load_center_offset_px():
    """Return the saved center offset, or the uncalibrated default."""
    try:
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            value = json.load(config_file).get("center_offset_px", 0)
        return int(value)
    except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError,
            TypeError, ValueError):
        return 0


def load_roi_right_frac():
    """Return the saved ROI right boundary, or the full-frame default."""
    try:
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            value = json.load(config_file).get("roi_right_frac", 1.0)
        value = float(value)
        return value if 0 < value <= 1 else 1.0
    except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError,
            TypeError, ValueError):
        return 1.0


def load_roi_left_frac():
    """Return the saved ROI left boundary, or the no-crop default."""
    try:
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            value = json.load(config_file).get("roi_left_frac", 0.0)
        value = float(value)
        return value if 0 <= value < 1 else 0.0
    except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError,
            TypeError, ValueError):
        return 0.0


def save_calibration(center_offset_px, roi_right_frac, roi_left_frac=0.0):
    """Atomically save all calibration values so a restart never sees a partial file."""
    center_value = int(center_offset_px)
    roi_value = float(roi_right_frac)
    left_roi_value = float(roi_left_frac)
    with NamedTemporaryFile("w", encoding="utf-8", dir=CONFIG_PATH.parent,
                            delete=False) as config_file:
        json.dump({"center_offset_px": center_value,
                   "roi_right_frac": roi_value,
                   "roi_left_frac": left_roi_value}, config_file, indent=2)
        config_file.write("\n")
        temporary_path = config_file.name
    os.replace(temporary_path, CONFIG_PATH)
    return center_value, roi_value, left_roi_value


def save_center_offset_px(center_offset_px):
    """Compatibility wrapper that preserves the currently saved ROI boundary."""
    return save_calibration(center_offset_px, load_roi_right_frac(),
                            load_roi_left_frac())[0]
