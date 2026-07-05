"""校验 VMD: 把 VMD 导回 PMX, 与源重定向 FBX 逐帧比较关键骨世界坐标。"""
import bpy, sys, os, addon_utils
from mathutils import Matrix

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
PMX, FBX, VMD = os.path.abspath(argv[0]), os.path.abspath(argv[1]), os.path.abspath(argv[2])

addon_utils.enable("bl_ext.user_default.mmd_tools", default_set=True, persistent=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)


def newo(b):
    return set(bpy.data.objects) - b


def arm_of(o):
    a = [x for x in o if x.type == "ARMATURE"]
    return a[0] if a else None


# PMX + VMD
b = set(bpy.data.objects)
bpy.ops.mmd_tools.import_model(filepath=PMX, scale=1.0, types={"ARMATURE"}, clean_model=False)
M = arm_of(newo(b))
bpy.ops.object.select_all(action="DESELECT")
M.select_set(True)
bpy.context.view_layer.objects.active = M
bpy.ops.mmd_tools.import_vmd(filepath=VMD, scale=1.0)

# source FBX
b = set(bpy.data.objects)
bpy.ops.import_scene.fbx(filepath=FBX)
F = arm_of(newo(b))


def hw_rest(arm, n):
    bn = arm.data.bones.get(n)
    return arm.matrix_world @ bn.head_local if bn else None


# align F to M (scale + translate, same as export)
k = (hw_rest(M, "足.L") - hw_rest(M, "足首.L")).length / (hw_rest(F, "足.L") - hw_rest(F, "足首.L")).length
F.matrix_world = Matrix.Diagonal((k, k, k, 1.0)) @ F.matrix_world
bpy.context.view_layer.update()
t = hw_rest(M, "足首.L") - hw_rest(F, "足首.L")
F.matrix_world = Matrix.Translation(t) @ F.matrix_world
bpy.context.view_layer.update()


def hw_pose(arm, n):
    pb = arm.pose.bones.get(n)
    return arm.matrix_world @ pb.head if pb else None


bones = ["頭", "手首.R", "手首.L", "ひじ.R", "足首.L", "足首.R", "腰"]
maxd = 0.0
for fr in [1, 45, 90, 135, 180]:
    bpy.context.scene.frame_set(fr)
    bpy.context.view_layer.update()
    diffs = []
    for n in bones:
        pm, pf = hw_pose(M, n), hw_pose(F, n)
        if pm and pf:
            d = (pm - pf).length
            diffs.append(d)
            maxd = max(maxd, d)
    print("frame %3d  max_bone_diff=%.4f  mean=%.4f" % (fr, max(diffs), sum(diffs) / len(diffs)))

print("OVERALL max bone world diff = %.4f (model height ~15 units)" % maxd)
