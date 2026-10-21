"""GeoCam Day 1 — GeoJSON -> Blender scene -> RGB image.

Run with Blender (not standard Python):
    blender --background --python-exit-code 1 --python main.py

Tested for syntax with CPython. Designed for Blender 4.5 LTS and 5.x.
"""

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


def find_project_root():
    """Locate sample_map.geojson for CLI and Blender Text Editor runs."""
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    for text_block in bpy.data.texts:
        if text_block.filepath and Path(text_block.filepath).name == "main.py":
            candidates.append(Path(bpy.path.abspath(text_block.filepath)).resolve().parent)
    candidates.append(Path.cwd())
    for candidate in candidates:
        if (candidate / "sample_map.geojson").is_file():
            return candidate
    raise FileNotFoundError("Keep main.py and sample_map.geojson in the same folder.")


ROOT = find_project_root()
MAP_FILE = ROOT / "sample_map.geojson"
OUTPUT = ROOT / "output"
EARTH_RADIUS_M = 6_371_000.0


def lonlat_to_xy(lon, lat, origin_lon, origin_lat):
    """Local east/north meters for a SMALL area near the origin.

    Equirectangular approximation, not a full geographic projection.
    """
    east = EARTH_RADIUS_M * math.radians(lon - origin_lon) * math.cos(math.radians(origin_lat))
    north = EARTH_RADIUS_M * math.radians(lat - origin_lat)
    return east, north


def make_material(name, rgb):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*rgb, 1.0)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.83
    return material


def make_box(name, location, dimensions, material, class_id=0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    obj["class_id"] = class_id  # Used later for segmentation ground truth
    obj.pass_index = class_id
    return obj


def make_building(name, ring, height, origin, material):
    # Polygon first ring: [(lon, lat), ...], last == first for a closed ring.
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        raise ValueError(f"Invalid building polygon: {name}")

    footprint = [lonlat_to_xy(lon, lat, *origin) for lon, lat in ring]
    count = len(footprint)
    vertices = [(x, y, 0.0) for x, y in footprint]
    vertices += [(x, y, height) for x, y in footprint]
    faces = [tuple(reversed(range(count))), tuple(range(count, count * 2))]
    for i in range(count):
        j = (i + 1) % count
        faces.append((i, j, j + count, i + count))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    obj["class_id"] = 2
    obj.pass_index = 2
    return obj


def make_road(name, coordinates, origin, width, material):
    # Each consecutive pair becomes one oriented rectangular road segment.
    points = [lonlat_to_xy(lon, lat, *origin) for lon, lat in coordinates]
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(points, points[1:])):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 0.01:
            continue
        obj = make_box(
            f"{name}_{i + 1}",
            ((x0 + x1) / 2, (y0 + y1) / 2, 0.045),
            (length, width, 0.09),
            material,
            class_id=1,
        )
        obj.rotation_euler[2] = math.atan2(dy, dx)


def make_camera():
    bpy.ops.object.camera_add(location=(45.0, -68.0, 72.0))
    camera = bpy.context.object
    camera.name = "GeoCam_Camera"
    target = Vector((0.0, 0.0, 4.0))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "PERSP"  # Pinhole / perspective camera
    camera.data.lens = 35.0
    bpy.context.scene.camera = camera
    return camera


def setup_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene
    # EEVEE engine identifier changed with Blender 5.0.
    scene.render.engine = "BLENDER_EEVEE" if bpy.app.version >= (5, 0, 0) else "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.filepath = str(OUTPUT / "rgb_day1.png")
    scene.render.image_settings.color_mode = "RGB"

    world = bpy.data.worlds.new("GeoCam_World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.62, 0.76, 0.94, 1.0)
        background.inputs["Strength"].default_value = 0.65

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 30))
    sun = bpy.context.object
    sun.name = "Sun"
    sun.rotation_euler = (math.radians(25), math.radians(-30), math.radians(-20))
    sun.data.energy = 2.0


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    geojson = json.loads(MAP_FILE.read_text(encoding="utf-8"))
    origin = tuple(geojson["properties"]["origin_lonlat"])
    setup_scene()

    grass = make_material("Grass", (0.21, 0.38, 0.25))
    road_mat = make_material("Road", (0.11, 0.13, 0.16))
    building_colors = [(0.85, 0.52, 0.35), (0.34, 0.57, 0.78), (0.87, 0.75, 0.43)]
    building_mats = [make_material(f"Building_{i}", rgb) for i, rgb in enumerate(building_colors, 1)]

    make_box("Ground", (0, 0, -0.15), (112, 104, 0.3), grass, class_id=0)
    building_count = 0
    road_count = 0

    for feature in geojson["features"]:
        geom = feature["geometry"]
        props = feature.get("properties", {})
        name = props.get("name", "Feature")
        if props.get("kind") == "building" and geom["type"] == "Polygon":
            material = building_mats[building_count % len(building_mats)]
            make_building(name, geom["coordinates"][0], float(props.get("height_m", 12)), origin, material)
            building_count += 1
        elif props.get("kind") == "road" and geom["type"] == "LineString":
            make_road(name, geom["coordinates"], origin, float(props.get("width_m", 8)), road_mat)
            road_count += 1

    camera = make_camera()
    blend_path = OUTPUT / "scene_day1.blend"
    # Avoid creating scene_day1.blend1 on subsequent runs.
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.render.render(write_still=True)

    print("\n[GeoCam Day 1] SUCCESS")
    print(f"Blender version : {bpy.app.version_string}")
    print(f"Map origin      : lon={origin[0]}, lat={origin[1]}")
    print(f"Created objects : {building_count} buildings, {road_count} road")
    print(f"Camera          : {tuple(round(v, 1) for v in camera.location)}")
    print(f"RGB image       : {OUTPUT / 'rgb_day1.png'}")
    print(f"Blender scene   : {blend_path}")


if __name__ == "__main__":
    main()
