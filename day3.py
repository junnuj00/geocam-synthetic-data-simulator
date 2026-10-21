"""GeoCam DAY 3 — Blender 4.5: three camera views, pinhole, noise, labels.

Put this alongside the *working DAY 2* main.py, day2.py, sample_map.geojson,
plus sensor_math.py. Run from Blender > Scripting > Open > day3.py > Alt+P.

Reuses DAY 1 Scene and proven DAY 2 Cycles IndexOB + Depth passes.
Output is written under output_day3/view_01 ... view_03.
"""
import importlib.util
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


def get_project_dir():
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    for text in bpy.data.texts:
        if text.filepath and Path(text.filepath).name == "day3.py":
            candidates.append(Path(bpy.path.abspath(text.filepath)).resolve().parent)
    candidates.append(Path.cwd())
    for folder in candidates:
        if all((folder / filename).is_file() for filename in
               ("main.py", "day2.py", "sample_map.geojson", "sensor_math.py")):
            return folder
    raise FileNotFoundError("Keep day3.py, day2.py, main.py, sample_map.geojson, sensor_math.py together")


ROOT = get_project_dir()
OUT_ROOT = ROOT / "output_day3"
OUT_ROOT.mkdir(exist_ok=True)


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(ROOT / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


day2 = load_module("day2.py", "geocam_day2_helpers")
sensor = load_module("sensor_math.py", "geocam_sensor_math")

# All views point at the same part of the DAY 1 scene. These are camera
# POSITIONS, not three cameras: a single camera is moved between renders.
CAMERA_VIEWS = [
    (45.0, -68.0, 72.0),
    (-55.0, -55.0, 68.0),
    (10.0, 75.0, 62.0),
]
LOOK_AT = Vector((0.0, 0.0, 4.0))


def configure_camera(scene, camera, location):
    camera.location = location
    camera.rotation_euler = (LOOK_AT - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "PERSP"
    camera.data.lens = 35.0
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.shift_x = 0.0
    camera.data.shift_y = 0.0
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()


def pinhole_metadata(scene, camera):
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    K = sensor.intrinsics(width, height, camera.data.lens, camera.data.sensor_width)

    # An OFF-CENTER 3D point makes the focal-length check meaningful.
    p_world = Vector((8.0, 3.0, 5.0))
    p_local = camera.matrix_world.inverted() @ p_world
    u, v = sensor.project_camera_point(p_local.x, p_local.y, -p_local.z, K)

    # Blender's own world_to_camera_view is an independent cross-check.
    ref = world_to_camera_view(scene, camera, p_world)
    u_ref = ref.x * width
    v_ref = (1.0 - ref.y) * height
    diff = math.hypot(u - u_ref, v - v_ref)
    if not ref.z > 0 or diff > 1.5:
        raise RuntimeError(f"Pinhole projection mismatch: {diff:.3f} px")

    # Matrix from world coordinates to Blender camera-local coordinates.
    world_to_camera = [[round(float(v), 8) for v in row]
                       for row in camera.matrix_world.inverted()]
    return {
        "model": "pinhole",
        "K": [[K["fx"], 0.0, K["cx"]],
              [0.0, K["fy"], K["cy"]],
              [0.0, 0.0, 1.0]],
        "position_world_m": [float(x) for x in camera.location],
        "world_to_camera_blender_4x4": world_to_camera,
        "lens_mm": camera.data.lens,
        "sensor_width_mm": camera.data.sensor_width,
        "image_size_wh": [width, height],
        "projection_test": {
            "world_point_xyz_m": [float(x) for x in p_world],
            "projected_pixel_uv": [round(u, 4), round(v, 4)],
            "blender_reference_uv": [round(u_ref, 4), round(v_ref, 4)],
            "error_pixels": round(diff, 6)
        }
    }


def add_noisy_depth(frame_dir, depth_exr, seed):
    width, height, exact = day2.read_exr_first_channel(depth_exr)
    noisy = sensor.add_depth_noise(exact, seed=seed)
    sensor.save_npy_f32(frame_dir / "depth_noisy.npy", noisy, width, height)
    preview = bytearray()
    for d in noisy:
        if not math.isfinite(d) or d <= 0:
            preview.append(0)  # invalid / missing
        else:
            preview.append(round(255 * (1.0 - min(d, 150.0) / 150.0)))
    day2.write_png(frame_dir / "depth_noisy_preview.png", width, height, preview)
    return sum(not math.isfinite(d) for d in noisy)


def render_one_view(scene, camera, position, view_no):
    frame_dir = OUT_ROOT / f"view_{view_no:02d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("depth_*.exr", "classid_*.exr"):
        for stale in frame_dir.glob(pattern):
            stale.unlink()

    configure_camera(scene, camera, position)
    cam_metadata = pinhole_metadata(scene, camera)

    # Reuse the DAY 2 compositor. Only its output folder changes.
    day2.OUTPUT = frame_dir
    day2.setup_compositor(scene)
    scene.render.filepath = str(frame_dir / "rgb.png")
    bpy.ops.render.render(write_still=True)

    for prefix, name in (("depth", "depth.exr"), ("classid", "class_ids.exr")):
        src = day2.pick_exr(prefix)
        dest = frame_dir / name
        if dest.exists():
            dest.unlink()
        src.rename(dest)

    width, height, counts = day2.create_segmentation_png(frame_dir / "class_ids.exr")
    day2.create_depth_preview(frame_dir / "depth.exr")
    invalid_pixels = add_noisy_depth(frame_dir, frame_dir / "depth.exr", seed=2026 + view_no)

    metadata = {
        "frame_id": view_no,
        "rgb": "rgb.png",
        "depth_gt": "depth.exr",
        "depth_noisy": "depth_noisy.npy",
        "semantic_mask": "segmentation_ids.png",
        "classes": {"0": "ground_or_background", "1": "road", "2": "building"},
        "pixel_counts": {str(k): v for k, v in counts.items()},
        "camera": cam_metadata,
        "depth_noise": {
            "type": "illustrative independent Gaussian + missing readings",
            "base_sigma_m": 0.05,
            "sigma_per_meter": 0.001,
            "dropout_probability": 0.01,
            "seed": 2026 + view_no,
            "invalid_or_missing_pixel_count": invalid_pixels,
            "unit": "m",
            "meaning": "Blender Z pass: camera forward-axis depth, not radial range"
        },
        "resolution_wh": [width, height]
    }
    (frame_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[GeoCam DAY 3] view_{view_no:02d} saved: RGB / depth / segmentation / noisy depth")
    print(f"[GeoCam DAY 3] projection error: {cam_metadata['projection_test']['error_pixels']:.6f} pixels")


def main():
    print("\n[GeoCam DAY 3] Reusing the successful DAY 1 + DAY 2 pipeline...")
    scene = day2.build_scene_using_day1()
    camera = scene.camera
    for i, position in enumerate(CAMERA_VIEWS, start=1):
        render_one_view(scene, camera, position, i)
    (OUT_ROOT / "README_results.txt").write_text(
        "Three views. Each includes RGB, depth EXR, segmentation mask, "
        "camera metadata, noisy depth .npy and previews.\n", encoding="utf-8")
    print("\n[GeoCam DAY 3] SUCCESS - all 3 viewpoints completed")
    print("Output directory:", OUT_ROOT)


if __name__ == "__main__":
    main()
