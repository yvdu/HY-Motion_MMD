"""仅看 VMD 驱动后的 M: 髋部是否在合理高度(不贴地)、比例是否正常。"""
import bpy, sys, os, addon_utils
argv = sys.argv; argv = argv[argv.index("--")+1:] if "--" in argv else []
PMX, VMD = os.path.abspath(argv[0]), os.path.abspath(argv[1])

addon_utils.enable("bl_ext.user_default.mmd_tools", default_set=True, persistent=True)
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
def arm_of(o): a=[x for x in o if x.type=="ARMATURE"]; return a[0] if a else None
b=set(bpy.data.objects)
bpy.ops.mmd_tools.import_model(filepath=PMX, scale=1.0, types={"ARMATURE"}, clean_model=False)
M=arm_of(set(bpy.data.objects)-b)
bpy.ops.object.select_all(action="DESELECT"); M.select_set(True); bpy.context.view_layer.objects.active=M
bpy.ops.mmd_tools.import_vmd(filepath=VMD, scale=1.0)

def rz(n):
    bn=M.data.bones.get(n); return (M.matrix_world@bn.head_local).z if bn else None
def pz(n):
    pb=M.pose.bones.get(n); return (M.matrix_world@pb.head).z if pb else None

print("REST   頭=%.3f 腰=%.3f 足首=%.3f  (body=%.3f, hip%%=%.0f%%)" % (
    rz("頭"), rz("腰"), rz("足首.L"), rz("頭")-rz("足首.L"),
    100*(rz("腰")-rz("足首.L"))/(rz("頭")-rz("足首.L"))))
for fr in [1,45,90,135,180]:
    bpy.context.scene.frame_set(fr); bpy.context.view_layer.update()
    body=pz("頭")-pz("足首.L")
    hippct=100*(pz("腰")-pz("足首.L"))/body if body else 0
    print("f%-4d  頭=%.3f 腰=%.3f 足首=%.3f  body=%.3f hip%%=%.0f%%" % (
        fr, pz("頭"), pz("腰"), pz("足首.L"), body, hippct))
