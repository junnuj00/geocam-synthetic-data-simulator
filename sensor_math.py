"""Pure-Python camera projection / toy depth-noise helpers (no bpy dependency)."""
import math
import random
import struct
import sys
from array import array
from pathlib import Path


def intrinsics(width, height, focal_length_mm, sensor_width_mm):
    """Blender perspective camera: HORIZONTAL sensor fit, square pixels, no lens shift."""
    if min(width, height) <= 0 or focal_length_mm <= 0 or sensor_width_mm <= 0:
        raise ValueError("Image dimensions, focal length and sensor width must be positive")
    focal_pixels = width * focal_length_mm / sensor_width_mm
    return {"fx": focal_pixels, "fy": focal_pixels,
            "cx": width / 2.0, "cy": height / 2.0,
            "width": width, "height": height}


def project_camera_point(x_right, y_up, z_forward, K):
    """Project a point in right/up/forward coordinates to (u,v) pixels (origin top-left)."""
    if z_forward <= 0:
        raise ValueError("Point is behind camera or lies on the image plane")
    u = K["fx"] * x_right / z_forward + K["cx"]
    v = -K["fy"] * y_up / z_forward + K["cy"]
    return u, v


def add_depth_noise(depths, seed=123, base_sigma_m=0.05,
                    sigma_per_meter=0.001, dropout_probability=0.01):
    """Illustrative camera-Z noise only; NOT a calibrated sensor model.

    Valid depths are perturbed by normal(0, base_sigma_m + sigma_per_meter * depth).
    Missing readings are NaN. Original depth array must remain unchanged.
    """
    if base_sigma_m < 0 or sigma_per_meter < 0 or not 0 <= dropout_probability <= 1:
        raise ValueError("Invalid noise parameters")
    rng = random.Random(seed)
    noisy = []
    for depth in depths:
        if not math.isfinite(depth) or depth <= 0:
            noisy.append(float("nan"))
        elif rng.random() < dropout_probability:
            noisy.append(float("nan"))
        else:
            sigma = base_sigma_m + sigma_per_meter * depth
            noisy.append(max(0.00001, depth + rng.gauss(0.0, sigma)))
    return noisy


def save_npy_f32(path, values_bottom_up, width, height):
    """Write standards-compatible NumPy .npy v1.0, float32, top-down HxW.

    Blender-loaded EXR pixels are bottom-up. The saved array and PNG masks
    share a top-left origin, so the numeric depths line up pixel-for-pixel.
    """
    if len(values_bottom_up) != width * height:
        raise ValueError("Depth array size does not match image dimensions")
    header = ("{'descr': '<f4', 'fortran_order': False, "
              f"'shape': ({height}, {width}), }}").encode("ascii")
    pad = (16 - ((10 + len(header) + 1) % 16)) % 16
    header = header + b" " * pad + b"\n"
    if len(header) > 65535:
        raise ValueError("NumPy header too large")
    flat_top_down = array("f")
    for y in range(height - 1, -1, -1):
        flat_top_down.extend(values_bottom_up[y * width:(y + 1) * width])
    if sys.byteorder != "little":
        flat_top_down.byteswap()
    with Path(path).open("wb") as file:
        file.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)))
        file.write(header)
        flat_top_down.tofile(file)
