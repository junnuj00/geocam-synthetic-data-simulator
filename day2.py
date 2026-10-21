"""GeoCam DAY 2: render RGB, depth (float EXR), and semantic segmentation (ID PNG).

Run with Blender (not standard Python) from the Scripting tab using Run Script,
or use `blender --background --python-exit-code 1 --python day2.py`.

Designed for Blender 4.5 LTS. Includes Blender 5.x compositor API support.
The output is not dependent on extra pip packages.
"""

import importlib.util
import json
import math
import struct
import zlib
from pathlib import Path

import bpy


def find_root():
    paths = []
    if "__file__" in globals():
        paths.append(Path(__file__).resolve().parent)
    for datablock in bpy.data.texts:
        if datablock.filepath and Path(datablock.filepath).name == "day2.py":
            paths.append(Path(bpy.path.abspath(datablock.filepath)).resolve().parent)
    paths.append(Path.cwd())
    for root in paths:
        if (root / "main.py").is_file() and (root / "sample_map.geojson").is_file():
            return root
    raise FileNotFoundError("Keep day2.py, main.py and sample_map.geojson in one folder.")


ROOT = find_root()
OUTPUT = ROOT / "output_day2"
OUTPUT.mkdir(parents=True, exist_ok=True)


def build_scene_using_day1():
    """Reuse the proven Day 1 scene generator, without modifying its source."""
    spec = importlib.util.spec_from_file_location("geocam_day1_main", str(ROOT / "main.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    return bpy.context.scene


def setup_compositor(scene):
    """Configure exact-data render passes with Blender 4.5 / 5.x APIs."""
    # Blender 4.5 EEVEE does NOT provide Object Index / IndexOB.
    # Cycles provides both Z (Depth) and IndexOB ground-truth passes.
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16  # Small scene: keep preview rendering fast.
    print("[GeoCam DAY 2] Engine: Cycles (Depth + IndexOB enabled)")
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True

    if bpy.app.version >= (5, 0, 0):
        tree = bpy.data.node_groups.new("GeoCam_Day2_Compositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
        tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        tree.nodes.clear()
        result = tree.nodes.new("CompositorNodeRLayers")
        group_output = tree.nodes.new("NodeGroupOutput")
        tree.links.new(result.outputs["Image"], group_output.inputs["Image"])
    else:
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        result = tree.nodes.new("CompositorNodeRLayers")
        composite = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(result.outputs["Image"], composite.inputs["Image"])

    def float_file_output(prefix, pass_socket):
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.label = prefix
        node.format.file_format = "OPEN_EXR"
        node.format.color_mode = "BW"
        node.format.color_depth = "32"
        node.format.exr_codec = "ZIP"
        if bpy.app.version >= (5, 0, 0):
            node.directory = str(OUTPUT)
            node.file_name = prefix + "_"
        else:
            node.base_path = str(OUTPUT)
            node.file_slots[0].path = prefix + "_"
        tree.links.new(result.outputs[pass_socket], node.inputs[0])
        return node

    float_file_output("depth", "Depth")
    float_file_output("classid", "IndexOB")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(OUTPUT / "rgb_day2.png")
    scene.frame_set(1)


def pick_exr(prefix):
    files = sorted(OUTPUT.glob(prefix + "_*.exr"))
    if not files:
        raise FileNotFoundError(f"No {prefix} pass saved. Check compositor settings: {OUTPUT}")
    if len(files) > 1:
        raise RuntimeError(f"Unexpected extra {prefix} passes. Clear output_day2 and try again.")
    return files[0]


def png_chunk(chunk_type, data):
    payload = chunk_type + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def write_png(path, width, height, values, rgb=False):
    """Minimal lossless PNG writer; Blender/python extra packages not needed.

    Values are bottom-up (like bpy.data.images Image.pixels). PNG rows are top-down.
    """
    stride = width * (3 if rgb else 1)
    assert len(values) == height * stride
    scanlines = bytearray()
    for row in range(height - 1, -1, -1):
        scanlines.append(0)  # no PNG filter
        scanlines.extend(values[row * stride: (row + 1) * stride])
    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2 if rgb else 0, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(bytes(scanlines), 6))
    png += png_chunk(b"IEND", b"")
    path.write_bytes(png)


def read_exr_first_channel(path):
    """Read unmodified floating-point EXR pixel data through Blender."""
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        channels = image.channels
        # Blender version/decoder may provide 1, 3, or 4 channels.
        pixel_data = image.pixels[:]
        values = [pixel_data[p] for p in range(0, len(pixel_data), channels)]
        if len(values) != width * height:
            raise RuntimeError("EXR pixel count does not match image dimensions")
        return width, height, values
    finally:
        bpy.data.images.remove(image)


def create_segmentation_png(src):
    width, height, values = read_exr_first_channel(src)
    # Strictly preserve class IDs 0=background/ground, 1=road, 2=building.
    label_ids = []
    for raw in values:
        if not math.isfinite(raw) or abs(raw - round(raw)) > 0.01:
            raise ValueError(f"IndexOB contains a non-integer class value: {raw}")
        label = int(round(raw))
        if label not in (0, 1, 2):
            raise ValueError(f"Unexpected class ID {label}; expected 0,1,2")
        label_ids.append(label)

    write_png(OUTPUT / "segmentation_ids.png", width, height, bytes(label_ids))
    colors = {0: (87, 145, 84), 1: (45, 48, 57), 2: (235, 142, 64)}
    preview = bytearray()
    for class_id in label_ids:
        preview.extend(colors[class_id])
    write_png(OUTPUT / "segmentation_preview.png", width, height, preview, rgb=True)
    return width, height, {value: label_ids.count(value) for value in sorted(set(label_ids))}


def create_depth_preview(src, limit_m=150.0):
    width, height, depths = read_exr_first_channel(src)
    gray = bytearray()
    for d in depths:
        if not math.isfinite(d) or d <= 0:
            gray.append(0)  # sky/no hit
        else:
            gray.append(int(round(255 * (1 - min(d, limit_m) / limit_m))))
    write_png(OUTPUT / "depth_preview.png", width, height, gray)
    return width, height


def main():
    print("\n[GeoCam DAY 2] Building the same DAY 1 scene...")
    scene = build_scene_using_day1()
    # Clean only this day's outputs, leaving Day 1 output intact.
    for prefix in ("depth_*.exr", "classid_*.exr"):
        for old in OUTPUT.glob(prefix):
            old.unlink()
    setup_compositor(scene)
    print("[GeoCam DAY 2] Rendering RGB + depth + object-index passes...")
    bpy.ops.render.render(write_still=True)

    depth_exr = pick_exr("depth")
    id_exr = pick_exr("classid")
    # Rename to stable, readable filenames.
    for src, new_name in ((depth_exr, "depth.exr"), (id_exr, "class_ids.exr")):
        dst = OUTPUT / new_name
        if dst.exists():
            dst.unlink()
        src.rename(dst)

    width, height, counts = create_segmentation_png(OUTPUT / "class_ids.exr")
    create_depth_preview(OUTPUT / "depth.exr")
    metadata = {
        "resolution": [width, height],
        "classes": {"0": "background_or_ground", "1": "road", "2": "building"},
        "camera_location_blender_m": [round(float(x), 6) for x in scene.camera.location],
        "lens_mm": float(scene.camera.data.lens),
        "files": {
            "rgb": "rgb_day2.png",
            "depth": "depth.exr",
            "segmentation_label_ids": "segmentation_ids.png",
            "segmentation_preview": "segmentation_preview.png"
        },
        "pixel_count_by_class": {str(k): v for k, v in counts.items()},
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[GeoCam DAY 2] SUCCESS")
    print("Output directory:", OUTPUT)
    print("RGB                 : rgb_day2.png")
    print("Depth (real values) : depth.exr")
    print("Depth preview       : depth_preview.png")
    print("Class IDs (raw EXR) : class_ids.exr")
    print("Class IDs (PNG)     : segmentation_ids.png")
    print("Mask preview        : segmentation_preview.png")
    print("Camera metadata     : metadata.json")
    print("Class pixel counts  :", counts)


if __name__ == "__main__":
    main()
