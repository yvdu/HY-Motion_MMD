"""Blender 脚本：导入 PMX + VMD，渲染 PNG 帧序列。

输出目录由第 3 个参数指定（写入 frame_XXXX.png）。
MP4 合成由 app.py 在 Blender 外完成（Blender 5 内嵌 Python 常无 ffmpeg/imageio）。

用法:
  blender -b --python scripts/render_pmx_vmd.py -- <model.pmx> <motion.vmd> <frames_dir>
"""
from __future__ import annotations

import math
import os
import sys

import addon_utils
import bpy
from mathutils import Vector

argv = sys.argv
argv = argv[argv.index("--") + 1 :] if "--" in argv else []
if len(argv) < 3:
    raise SystemExit("usage: blender -b --python render_pmx_vmd.py -- <pmx> <vmd> <frames_dir>")

PMX = os.path.abspath(argv[0])
VMD = os.path.abspath(argv[1])
FRAME_DIR = os.path.abspath(argv[2])
os.makedirs(FRAME_DIR, exist_ok=True)

addon_utils.enable("bl_ext.user_default.mmd_tools", default_set=True, persistent=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.ops.mmd_tools.import_model(filepath=PMX, scale=1.0, clean_model=False)
arm = None
for obj in bpy.data.objects:
    if obj.type == "ARMATURE":
        arm = obj
        break
if arm is None:
    raise RuntimeError("PMX import failed: no armature")

bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.mmd_tools.import_vmd(filepath=VMD, scale=1.0)

scene = bpy.context.scene
fs, fe = 1, 30
if arm.animation_data and arm.animation_data.action:
    fr = arm.animation_data.action.frame_range
    fs, fe = int(fr[0]), int(fr[1])
scene.frame_start = fs
scene.frame_end = max(fe, fs + 1)
scene.frame_set(fs)

bpy.context.view_layer.update()
min_v = Vector((1e9, 1e9, 1e9))
max_v = Vector((-1e9, -1e9, -1e9))
has_mesh = False
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    has_mesh = True
    for corner in obj.bound_box:
        w = obj.matrix_world @ Vector(corner)
        min_v = Vector((min(min_v.x, w.x), min(min_v.y, w.y), min(min_v.z, w.z)))
        max_v = Vector((max(max_v.x, w.x), max(max_v.y, w.y), max(max_v.z, w.z)))
if not has_mesh:
    min_v, max_v = Vector((-1, -1, 0)), Vector((1, 1, 2))

center = (min_v + max_v) * 0.5
height = max(max_v.z - min_v.z, 1.0)
radius = max((max_v - min_v).length * 0.55, height * 0.8)

bpy.ops.object.light_add(type="SUN", location=(center.x, center.y - radius, center.z + height))
bpy.context.active_object.data.energy = 3.0
bpy.ops.object.light_add(type="AREA", location=(center.x + radius, center.y, center.z + height * 0.5))
bpy.context.active_object.data.energy = 200.0

bpy.ops.object.camera_add()
cam = bpy.context.active_object
cam.location = (center.x, center.y - radius * 1.6, center.z + height * 0.35)
direction = center - cam.location
cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
cam.rotation_euler.x = max(cam.rotation_euler.x, math.radians(75))
scene.camera = cam

bpy.ops.mesh.primitive_plane_add(size=radius * 6, location=(center.x, center.y, min_v.z))
ground = bpy.context.active_object
mat = bpy.data.materials.new("Ground")
mat.diffuse_color = (0.85, 0.85, 0.88, 1.0)
ground.data.materials.append(mat)

try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except Exception:
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except Exception:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 16

scene.render.fps = 30
scene.render.resolution_x = 960
scene.render.resolution_y = 544  # 16 的倍数，便于 H.264 编码
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.filepath = os.path.join(FRAME_DIR, "frame_")

if scene.world is None:
    scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.92, 0.94, 0.98, 1.0)
    bg.inputs[1].default_value = 1.0

print(f">>> Rendering frames {fs}-{fe} -> {FRAME_DIR}")
bpy.ops.render.render(animation=True)
print(f">>> DONE frames: {FRAME_DIR}")
