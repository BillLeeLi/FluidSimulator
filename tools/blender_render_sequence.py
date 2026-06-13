#!/usr/bin/env python3
"""Render FluidSimulator offline-export folders with Blender/Cycles.

Design goals:
  * Keep the boat scene, but make the water/glass look closer to PBRT-style offline rendering.
  * Never let a full white glass wall block the camera by default.
  * Use exported Blender Z-up PLYs directly; do not flip/rotate the boat in the renderer.
  * Auto-frame the real PLY bounds instead of relying on a hard-coded camera.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Iterable, Optional, Sequence, Tuple

try:
    import bpy
    from mathutils import Vector
except Exception as exc:  # pragma: no cover - only runs inside Blender
    raise SystemExit("This script must be run by Blender: blender -b -P tools/blender_render_sequence.py -- ...") from exc

Vec3 = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Render FluidSimulator offline PLY frames in Blender.")
    p.add_argument("--input", "--root", dest="root", required=True, help="Folder created by xmake run offline-export")
    p.add_argument("--output", required=True, help="Render output folder")
    p.add_argument("--render-dir", default="frames", help="Subfolder inside --output for PNG frames")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="Exclusive end frame. Default: all detected frames")
    p.add_argument("--frames", type=int, default=None, help="Render this many frames from --start")
    p.add_argument("--frame", type=int, default=None, help="Render only one frame")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--resolution-x", type=int, default=1280)
    p.add_argument("--resolution-y", type=int, default=720)
    p.add_argument("--gpu", action="store_true", help="Use GPU/Cycles if available")
    p.add_argument("--mode", choices=["surface", "metaball", "auto"], default="surface")
    p.add_argument("--tank-style", choices=["frame", "open-glass", "glass", "none"], default="frame")
    p.add_argument("--camera", choices=["boat", "front", "side", "top", "threequarter"], default="front")
    p.add_argument("--camera-scan", type=int, default=12, help="How many frames to scan for auto camera bounds")
    p.add_argument("--ortho-scale", type=float, default=0.0, help="Override orthographic scale")
    p.add_argument("--camera-distance", type=float, default=0.0, help="Extra multiplier for camera distance; 0=auto")
    p.add_argument("--water-alpha", type=float, default=0.55)
    p.add_argument("--water-roughness", type=float, default=0.018)
    p.add_argument("--water-ior", type=float, default=1.333)
    p.add_argument("--subdivision", type=int, default=1, help="Optional subdivision level for fluid surface; default 1 for smoother offline water")
    p.add_argument("--mesh-merge-distance", type=float, default=1e-5, help="Weld duplicate vertices after PLY import; helps Marching Tetrahedra surfaces shade smoothly")
    p.add_argument("--metaball-radius", type=float, default=0.018)
    p.add_argument("--metaball-resolution", type=float, default=0.020)
    p.add_argument("--make-mp4", action="store_true", help="Also call make_video.py if available")
    return p.parse_args(argv)


def load_metadata(root: Path) -> dict:
    path = root / "metadata.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def frame_number_from_path(path: Path) -> Optional[int]:
    m = re.search(r"_(\d{6})", path.name)
    if not m:
        m = re.search(r"_(\d+)", path.name)
    return int(m.group(1)) if m else None


def available_frames(root: Path) -> list[int]:
    frames_dir = root / "frames"
    files = list(frames_dir.glob("fluid_*.ply"))
    nums = sorted(n for f in files if (n := frame_number_from_path(f)) is not None)
    return nums


def ply_bounds(path: Path) -> Optional[Tuple[Vector, Vector]]:
    """Fast ASCII/binary-tolerant PLY header + vertex scan for bounds.

    The exporter writes ASCII PLY. This parser intentionally only reads x/y/z from
    the vertex block so the camera can be framed before Blender imports anything.
    """
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            header_lines: list[str] = []
            vertex_count = 0
            fmt = "ascii"
            while True:
                line = f.readline()
                if not line:
                    return None
                s = line.decode("ascii", errors="ignore").strip()
                header_lines.append(s)
                if s.startswith("format"):
                    fmt = s.split()[1]
                if s.startswith("element vertex"):
                    vertex_count = int(s.split()[2])
                if s == "end_header":
                    break
            if vertex_count <= 0:
                return None
            if fmt != "ascii":
                return None

            mn = Vector((float("inf"), float("inf"), float("inf")))
            mx = Vector((float("-inf"), float("-inf"), float("-inf")))
            valid = False
            for _ in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                parts = line.decode("ascii", errors="ignore").split()
                if len(parts) < 3:
                    continue
                p = Vector((float(parts[0]), float(parts[1]), float(parts[2])))
                mn.x = min(mn.x, p.x); mn.y = min(mn.y, p.y); mn.z = min(mn.z, p.z)
                mx.x = max(mx.x, p.x); mx.y = max(mx.y, p.y); mx.z = max(mx.z, p.z)
                valid = True
            return (mn, mx) if valid else None
    except Exception:
        return None


def union_bounds(bounds: Iterable[Optional[Tuple[Vector, Vector]]]) -> Tuple[Vector, Vector]:
    mn = Vector((float("inf"), float("inf"), float("inf")))
    mx = Vector((float("-inf"), float("-inf"), float("-inf")))
    valid = False
    for b in bounds:
        if b is None:
            continue
        a, c = b
        mn.x = min(mn.x, a.x); mn.y = min(mn.y, a.y); mn.z = min(mn.z, a.z)
        mx.x = max(mx.x, c.x); mx.y = max(mx.y, c.y); mx.z = max(mx.z, c.z)
        valid = True
    if not valid:
        return Vector((-0.55, -0.55, -0.55)), Vector((0.55, 0.55, 0.55))
    pad = Vector((0.06, 0.06, 0.06))
    return mn - pad, mx + pad


def compute_scene_bounds(root: Path, frames: Sequence[int], scan_count: int) -> Tuple[Vector, Vector]:
    frames_dir = root / "frames"
    use_frames = frames[: max(1, min(len(frames), scan_count))]
    all_bounds: list[Optional[Tuple[Vector, Vector]]] = []
    for frame in use_frames:
        tag = f"{frame:06d}"
        all_bounds.append(ply_bounds(frames_dir / f"fluid_{tag}.ply"))
        for rigid in frames_dir.glob(f"rigid_{tag}_*.ply"):
            all_bounds.append(ply_bounds(rigid))
    # Always include the simulation tank, otherwise early frames with small water volumes get too tight.
    all_bounds.append((Vector((-0.5, -0.5, -0.5)), Vector((0.5, 0.5, 0.5))))
    return union_bounds(all_bounds)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def set_cycles(samples: int, use_gpu: bool) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.preview_samples = max(8, min(64, samples // 2))
    scene.cycles.use_denoising = True
    # Enough bounces for glass/water, but not excessive for classroom renders.
    scene.cycles.max_bounces = 12
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.transmission_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 8

    if use_gpu:
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            # Prefer OptiX, then CUDA. HIP warnings on NVIDIA machines are harmless.
            for backend in ("OPTIX", "CUDA", "HIP"):
                try:
                    prefs.compute_device_type = backend
                    prefs.get_devices()
                    devices = [d for d in prefs.devices if getattr(d, "type", "") != "CPU"]
                    if devices:
                        for d in prefs.devices:
                            d.use = True
                        scene.cycles.device = "GPU"
                        print(f"[BlenderRender] Using GPU compute backend: {backend}")
                        return
                except Exception:
                    continue
        except Exception as exc:
            print(f"[BlenderRender] GPU setup skipped: {exc}")
    scene.cycles.device = "CPU"
    print("[BlenderRender] Using CPU Cycles")


def set_color_management() -> None:
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def mat_principled(name: str, color=(1, 1, 1, 1), roughness=0.5, metallic=0.0) -> bpy.types.Material:
    m = bpy.data.materials.new(name)
    m.diffuse_color = color
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        def set_input(names, value):
            for n in names:
                if n in bsdf.inputs:
                    bsdf.inputs[n].default_value = value
                    return
        set_input(["Base Color"], color)
        set_input(["Roughness"], roughness)
        set_input(["Metallic"], metallic)
        set_input(["Alpha"], color[3])
    if hasattr(m, "use_screen_refraction"):
        m.use_screen_refraction = True
    return m


def mat_solid_diffuse(name: str, color=(0.35, 0.02, 0.02, 1.0), roughness=0.45) -> bpy.types.Material:
    """Opaque solid material for rigid bodies.

    This avoids Blender-version-dependent Principled BSDF input names and
    makes the rigid-body color deterministic in Cycles.
    """
    m = bpy.data.materials.new(name)
    m.diffuse_color = color
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
    if "Color" in diffuse.inputs:
        diffuse.inputs["Color"].default_value = color
    if "Roughness" in diffuse.inputs:
        diffuse.inputs["Roughness"].default_value = roughness
    nt.links.new(diffuse.outputs["BSDF"], out.inputs["Surface"])
    return m


def mat_water(alpha: float, roughness: float, ior: float) -> bpy.types.Material:
    m = bpy.data.materials.new("water_dielectric_with_absorption")
    m.diffuse_color = (0.62, 0.88, 1.0, 1.0)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    surface = nt.nodes.new("ShaderNodeBsdfPrincipled")
    absorption = nt.nodes.new("ShaderNodeVolumeAbsorption")

    def set_surface_input(names: Sequence[str], value) -> None:
        for name in names:
            if name in surface.inputs:
                surface.inputs[name].default_value = value
                return

    set_surface_input(["Base Color"], (0.72, 0.93, 1.0, 1.0))
    set_surface_input(["Roughness"], roughness)
    set_surface_input(["IOR"], ior)
    set_surface_input(["Transmission Weight", "Transmission"], 1.0)
    if "Alpha" in surface.inputs:
        surface.inputs["Alpha"].default_value = 1.0

    absorption.inputs["Color"].default_value = (0.35, 0.78, 0.95, 1.0)
    absorption.inputs["Density"].default_value = max(0.0, min(1.0, alpha)) * 0.6
    nt.links.new(surface.outputs["BSDF"], out.inputs["Surface"])
    nt.links.new(absorption.outputs["Volume"], out.inputs["Volume"])
    return m


def mat_glass_frame() -> bpy.types.Material:
    return mat_principled("tank_glass_frame", (0.78, 0.94, 1.0, 0.32), roughness=0.06)


def import_ply(path: Path, name: Optional[str] = None) -> Optional[bpy.types.Object]:
    if not path.exists():
        return None
    before = set(bpy.data.objects)
    try:
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    except Exception as exc:
        print(f"[BlenderRender] Failed to import {path}: {exc}")
        return None
    after = set(bpy.data.objects)
    new = list(after - before)
    if not new:
        obj = bpy.context.object
    else:
        obj = new[0]
    if obj is None:
        return None
    if name:
        obj.name = name
    return obj


def smooth_mesh(obj: bpy.types.Object, subdivision: int = 0, merge_distance: float = 1e-5) -> None:
    if obj is None or obj.type != "MESH":
        return
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # The C++ exporter currently emits independent triangle vertices.  Without
    # welding, Blender cannot interpolate normals across triangle borders and
    # the fluid shows marching-tetrahedra striping.
    if merge_distance > 0:
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            if hasattr(bpy.ops.mesh, "remove_doubles"):
                bpy.ops.mesh.remove_doubles(threshold=merge_distance)
            else:
                bpy.ops.mesh.merge_by_distance(distance=merge_distance)
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception as exc:
            print(f"[BlenderRender] vertex weld skipped for {obj.name}: {exc}")
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass

    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass
    obj.select_set(False)
    if subdivision > 0:
        mod = obj.modifiers.new("offline_subdivision", "SUBSURF")
        mod.levels = subdivision
        mod.render_levels = subdivision
    try:
        mod = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
        mod.keep_sharp = False
    except Exception:
        pass


def create_metaball_from_points(path: Path, material: bpy.types.Material, radius: float, resolution: float) -> Optional[bpy.types.Object]:
    b = ply_bounds(path)
    if b is None:
        return None
    # Read vertices. For performance, cap very dense point clouds with a stride.
    pts: list[Vector] = []
    with path.open("rb") as f:
        vertex_count = 0
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            if line == "end_header":
                break
        stride = max(1, vertex_count // 3500)
        for i in range(vertex_count):
            line = f.readline()
            if i % stride != 0:
                continue
            parts = line.decode("ascii", errors="ignore").split()
            if len(parts) >= 3:
                pts.append(Vector((float(parts[0]), float(parts[1]), float(parts[2]))))
    if not pts:
        return None
    mb = bpy.data.metaballs.new("fluid_metaball_data")
    mb.resolution = resolution
    mb.render_resolution = resolution * 0.75
    obj = bpy.data.objects.new("fluid_metaball", mb)
    bpy.context.collection.objects.link(obj)
    for p in pts:
        e = mb.elements.new(type="BALL")
        e.co = p
        e.radius = radius
        e.stiffness = 2.0
    obj.data.materials.append(material)
    return obj


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(bounds_min: Vector, bounds_max: Vector, mode: str, ortho_override: float, distance_override: float) -> bpy.types.Object:
    """Set up a standard Blender Z-up camera.

    Exported PLY coordinates are already converted by offline-export:
        sim (x, y_up, z_depth) -> blender (x, -z_depth, y_up)

    Therefore the camera and environment should use Blender's normal Z-up
    convention.  This is the main fix for the previous vertical-water result.
    """
    center = (bounds_min + bounds_max) * 0.5
    span = bounds_max - bounds_min
    radius = max(span.x, span.y, span.z, 0.6)

    if mode == "front":
        # Front view: centered on the tank front, with a slight upward tilt so the boat interior and waterline are both visible.
        direction = Vector((0.0, -1.0, 0.32))
    elif mode == "side":
        direction = Vector((1.0, -0.18, 0.34))
    elif mode == "top":
        direction = Vector((0.12, -0.18, 1.0))
    elif mode == "threequarter":
        direction = Vector((1.0, -1.0, 0.70))
    else:  # boat: 3/4 view from front-right-above; avoids the back pane.
        direction = Vector((1.10, -1.18, 0.72))
    direction.normalize()

    distance = radius * (2.8 if distance_override <= 0 else distance_override)
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = center + direction * distance
    # Aim around the boat/waterline, not the geometric center of tall spray.
    look_at(cam, center + Vector((0.0, 0.0, -0.05 * radius)))
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_override if ortho_override > 0 else max(span.x * 1.35, span.y * 1.35, span.z * 1.95, 1.05)
    cam_data.lens = 45
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    bpy.context.scene.camera = cam
    print(f"[BlenderRender] Camera location={tuple(round(v, 4) for v in cam.location)} ortho={cam_data.ortho_scale:.4f}")
    return cam

def add_area_light(name: str, location: Vec3, target: Vector, power: float, size: float) -> None:
    light_data = bpy.data.lights.new(name, type="AREA")
    light_data.energy = power
    light_data.size = size
    obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    look_at(obj, target)


def add_box(name: str, loc: Vec3, scale: Vec3, mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    return obj


def setup_environment(bounds_min: Vector, bounds_max: Vector, tank_style: str) -> None:
    """Create lights, floor, and tank in standard Blender Z-up coordinates."""
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.72, 0.80, 0.88)

    ground_mat = mat_principled("matte_warm_floor", (0.70, 0.70, 0.66, 1.0), roughness=0.68)
    glass_mat = mat_glass_frame()

    # Lights: z is vertical now. Area lights emit along local -Z, so aim them
    # at the exported scene instead of relying on their default rotation.
    light_target = (bounds_min + bounds_max) * 0.5
    add_area_light("large_softbox", (0.2, -1.6, 2.3), light_target, 580, 4.2)
    add_area_light("front_water_highlight", (-1.2, -0.9, 1.45), light_target, 95, 2.0)
    add_area_light("rim_light", (1.4, 1.2, 1.25), light_target, 65, 2.8)

    # Ground plane below the tank.  The old script used Y as vertical, which
    # was the reason the water/tank appeared rotated.  Here Z is vertical.
    ground_z = -0.535
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, ground_z - 0.012))
    ground = bpy.context.object
    ground.name = "matte_ground_under_tank"
    ground.dimensions = (1.55, 1.55, 0.024)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ground.data.materials.append(ground_mat)

    if tank_style == "none":
        return

    # Simulation tank inner bounds after exporter mapping are still [-0.5,0.5]^3,
    # but the vertical axis is now Blender Z.
    mn = Vector((-0.5, -0.5, -0.5))
    mx = Vector((0.5, 0.5, 0.5))
    thickness = 0.010
    xs = [mn.x, mx.x]
    ys = [mn.y, mx.y]
    zs = [mn.z, mx.z]

    # 12 edge rods.  Vertical rods run along Z, not Y.
    for x in xs:
        for y in ys:
            add_box("tank_vertical_edge", (x, y, 0.0), (thickness, thickness, 1.0), glass_mat)
    for x in xs:
        for z in zs:
            add_box("tank_depth_edge", (x, 0.0, z), (thickness, 1.0, thickness), glass_mat)
    for y in ys:
        for z in zs:
            add_box("tank_width_edge", (0.0, y, z), (1.0, thickness, thickness), glass_mat)

    if tank_style in {"open-glass", "glass"}:
        # Thin panes. open-glass omits the front pane; the camera looks from -Y.
        pane_mat = mat_principled("very_light_tank_pane", (0.80, 0.96, 1.0, 0.14), roughness=0.035)
        add_box("tank_bottom_pane", (0, 0, mn.z - 0.003), (1.0, 1.0, 0.006), pane_mat)
        add_box("tank_back_pane", (0, mx.y + 0.003, 0), (1.0, 0.006, 1.0), pane_mat)
        add_box("tank_left_pane", (mn.x - 0.003, 0, 0), (0.006, 1.0, 1.0), pane_mat)
        add_box("tank_right_pane", (mx.x + 0.003, 0, 0), (0.006, 1.0, 1.0), pane_mat)
        if tank_style == "glass":
            add_box("tank_front_pane", (0, mn.y - 0.003, 0), (1.0, 0.006, 1.0), pane_mat)

def remove_frame_objects() -> None:
    frame_data = []
    for obj in list(bpy.context.scene.objects):
        if obj.name.startswith(("fluid_", "rigid_", "frame_import_")) or obj.name == "fluid_metaball":
            if obj.type in {"MESH", "META"} and obj.data is not None:
                frame_data.append((obj.type, obj.data))
            bpy.data.objects.remove(obj, do_unlink=True)

    for data_type, data in frame_data:
        if data.users != 0:
            continue
        if data_type == "MESH":
            bpy.data.meshes.remove(data)
        elif data_type == "META":
            bpy.data.metaballs.remove(data)




def assign_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    """Force all mesh faces to use exactly this material.

    Some Blender PLY imports create an existing default/white material slot.
    If we only append a new material, faces may keep material_index=0 and
    continue rendering white. Clearing slots and resetting polygon indices makes
    the requested material deterministic.
    """
    if obj is None or obj.type != "MESH":
        return
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.material_index = 0
    obj.active_material = mat
    obj.color = mat.diffuse_color

def render_frame(args: argparse.Namespace, frame: int, mats: dict[str, bpy.types.Material], frames_dir: Path, render_dir: Path) -> None:
    remove_frame_objects()
    tag = f"{frame:06d}"

    fluid_obj = None
    fluid_ply = frames_dir / f"fluid_{tag}.ply"
    particle_ply = frames_dir / f"particles_{tag}.ply"
    if args.mode == "metaball":
        fluid_obj = create_metaball_from_points(particle_ply, mats["water"], args.metaball_radius, args.metaball_resolution)
    else:
        fluid_obj = import_ply(fluid_ply, name=f"fluid_surface_{tag}")
        if fluid_obj and fluid_obj.type == "MESH":
            assign_material(fluid_obj, mats["water"])
            smooth_mesh(fluid_obj, args.subdivision, args.mesh_merge_distance)
        if (fluid_obj is None or (getattr(fluid_obj.data, "vertices", []) and len(fluid_obj.data.vertices) == 0)) and args.mode == "auto":
            fluid_obj = create_metaball_from_points(particle_ply, mats["water"], args.metaball_radius, args.metaball_resolution)

    rigid_files = sorted(frames_dir.glob(f"rigid_{tag}_*.ply"))
    for i, path in enumerate(rigid_files):
        obj = import_ply(path, name=f"rigid_{tag}_{i:03d}")
        if obj is None:
            continue
        mat = mats["boat"] if "boat" in path.name.lower() or "kenney" in path.name.lower() else mats["rigid"]
        assign_material(obj, mat)
        smooth_mesh(obj, 0, args.mesh_merge_distance)

    bpy.context.scene.frame_set(frame)
    bpy.context.scene.render.filepath = str(render_dir / f"frame_{tag}.png")
    print(f"[BlenderRender] Rendering frame {tag}")
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    print(f"[BlenderRender] USING SCRIPT: {Path(__file__).resolve()}")
    root = Path(args.root).resolve()
    frames_dir = root / "frames"
    if not frames_dir.exists():
        raise SystemExit(f"No frames directory found: {frames_dir}")

    detected = available_frames(root)
    if not detected:
        raise SystemExit(f"No fluid_*.ply files found under {frames_dir}")

    if args.frame is not None:
        frames = [args.frame]
    else:
        start = args.start
        if args.frames is not None:
            end = start + args.frames
        elif args.end is not None:
            end = args.end
        else:
            end = max(detected) + 1
        frames = [f for f in detected if start <= f < end]
    if not frames:
        raise SystemExit("No frames selected for rendering")

    output = Path(args.output).resolve()
    render_dir = output / args.render_dir
    render_dir.mkdir(parents=True, exist_ok=True)

    bounds_min, bounds_max = compute_scene_bounds(root, frames, args.camera_scan)
    print(f"[BlenderRender] Auto bounds min={tuple(round(v, 4) for v in bounds_min)} max={tuple(round(v, 4) for v in bounds_max)}")

    clear_scene()
    bpy.context.scene.render.resolution_x = args.resolution_x
    bpy.context.scene.render.resolution_y = args.resolution_y
    bpy.context.scene.render.film_transparent = False
    set_cycles(args.samples, args.gpu)
    set_color_management()

    setup_environment(bounds_min, bounds_max, args.tank_style)
    setup_camera(bounds_min, bounds_max, args.camera, args.ortho_scale, args.camera_distance)

    mats = {
        "water": mat_water(args.water_alpha, args.water_roughness, args.water_ior),
        "boat": mat_solid_diffuse("boat_dark_red_forced", (0.45, 0.00, 0.00, 1.0), roughness=0.38),
        "rigid": mat_solid_diffuse("rigid_dark_red_forced", (0.45, 0.00, 0.00, 1.0), roughness=0.45),
    }
    print(f"[BlenderRender] rigid diffuse_color={tuple(round(v, 3) for v in mats['rigid'].diffuse_color)}")

    for frame in frames:
        render_frame(args, frame, mats, frames_dir, render_dir)

    print(f"[BlenderRender] Done. PNGs are in {render_dir}")


if __name__ == "__main__":
    main()
