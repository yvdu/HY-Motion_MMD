"""Headless PMX -> FBX converter (run inside Blender).

用法:
    blender --background --factory-startup --python pmx_to_fbx.py -- <input.pmx> <output.fbx> [mmd_tools.zip]

流程:
    1. 安装/启用开源插件 mmd_tools (blender_mmd_tools) —— Blender 不能原生读 PMX。
    2. 清空默认场景, 用 mmd_tools 导入 PMX (网格 + 骨架 + 蒙皮权重)。
    3. 导出 FBX (FBX202000, 不带 leaf bone, Y-up)。
"""
import os
import sys
import traceback

import bpy
import addon_utils


MODULE = "bl_ext.user_default.mmd_tools"


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(argv) < 2:
        raise SystemExit("需要参数: <input.pmx> <output.fbx> [mmd_tools.zip]")
    pmx = os.path.abspath(argv[0])
    fbx = os.path.abspath(argv[1])
    zip_path = os.path.abspath(argv[2]) if len(argv) > 2 else None
    return pmx, fbx, zip_path


def mmd_ready():
    return MODULE in bpy.context.preferences.addons


def ensure_mmd_tools(zip_path):
    # 1) 已安装则直接启用
    try:
        addon_utils.enable(MODULE, default_set=True, persistent=True)
    except Exception:
        pass
    if mmd_ready():
        return True

    # 2) 从 zip 安装为扩展 (Blender 4.2+ extensions 系统)
    if zip_path and os.path.isfile(zip_path):
        try:
            bpy.ops.extensions.package_install_files(
                repo="user_default", filepath=zip_path, enable_on_install=True,
            )
        except Exception:
            traceback.print_exc()
        try:
            addon_utils.enable(MODULE, default_set=True, persistent=True)
        except Exception:
            traceback.print_exc()
    return mmd_ready()


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def import_pmx(pmx_path):
    bpy.ops.mmd_tools.import_model(
        filepath=pmx_path,
        scale=1.0,
        types={"MESH", "ARMATURE", "MORPHS"},
        clean_model=True,
        remove_doubles=False,
        log_level="WARNING",
    )


def export_fbx(fbx_path):
    out_dir = os.path.dirname(fbx_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    bpy.ops.export_scene.fbx(
        filepath=fbx_path,
        use_selection=False,
        apply_unit_scale=True,
        global_scale=1.0,
        bake_space_transform=False,
        object_types={"ARMATURE", "MESH", "EMPTY"},
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        use_armature_deform_only=False,
        bake_anim=False,
        path_mode="COPY",
        embed_textures=True,
        axis_forward="-Z",
        axis_up="Y",
    )


def report():
    n_mesh = sum(1 for o in bpy.data.objects if o.type == "MESH")
    n_arm = sum(1 for o in bpy.data.objects if o.type == "ARMATURE")
    n_bones = 0
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            n_bones += len(o.data.bones)
    print("[report] meshes=%d armatures=%d bones=%d" % (n_mesh, n_arm, n_bones))


def main():
    pmx, fbx, zip_path = parse_args()
    if not os.path.isfile(pmx):
        raise SystemExit("找不到输入 PMX: %s" % pmx)

    print("[1/4] 启用 mmd_tools ...")
    if not ensure_mmd_tools(zip_path):
        raise SystemExit("mmd_tools 启用失败, 无法导入 PMX")

    print("[2/4] 清空场景 ...")
    clear_scene()

    print("[3/4] 导入 PMX: %s" % pmx)
    import_pmx(pmx)
    report()

    print("[4/4] 导出 FBX: %s" % fbx)
    export_fbx(fbx)

    if os.path.isfile(fbx):
        size_mb = os.path.getsize(fbx) / (1024.0 * 1024.0)
        print("DONE  %s  (%.2f MB)" % (fbx, size_mb))
    else:
        raise SystemExit("导出失败: 未生成 FBX")


if __name__ == "__main__":
    main()
