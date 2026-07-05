"""自定义 FBX->VMD 导出, 绕过 mmd_tools 导出转换器的 reflection 缺陷。

关键:
  - 复用 mmd_tools 导入侧的 BoneConverter 矩阵 q = to_quaternion(transpose(swapYZ(matrix_local)))
    (不做 invert), 解析地反演旋转: vmd_rot = q.conj() @ local_rot @ q  -> 与导入精确互逆。
  - 复用 mmd_tools 的 vmd 二进制序列化(头/打包/插值默认值)。
  - MMD 约定: 根位移放到可移动的 センター; 足部用 足ＩＫ/つま先ＩＫ 跟随重定向脚踝/脚尖。

用法: blender -b --python fbx_to_vmd_custom.py -- <model.pmx> <anim_bin.fbx> <out.vmd> [ref_vmd_for_interp]
"""
import bpy, sys, os, addon_utils, importlib
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv; argv = argv[argv.index("--")+1:] if "--" in argv else []
PMX, FBX, OUT = os.path.abspath(argv[0]), os.path.abspath(argv[1]), os.path.abspath(argv[2])
REF_VMD = os.path.abspath(argv[3]) if len(argv) > 3 else None

addon_utils.enable("bl_ext.user_default.mmd_tools", default_set=True, persistent=True)
vmd = importlib.import_module("bl_ext.user_default.mmd_tools.core.vmd")

bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
def newo(b): return set(bpy.data.objects)-b
def arm_of(o):
    a=[x for x in o if x.type=="ARMATURE"]; return a[0] if a else None

b=set(bpy.data.objects)
bpy.ops.mmd_tools.import_model(filepath=PMX, scale=1.0, types={"ARMATURE"}, clean_model=False)
M=arm_of(newo(b))
b=set(bpy.data.objects); bpy.ops.import_scene.fbx(filepath=FBX); F=arm_of(newo(b))
print("M.matrix_world identity:", M.matrix_world == Matrix.Identity(4))

# 源 FBX 的 root(全ての親) 被烘入了 ~0.01 的 spurious 缩放(cm/m 单位伪影),
# 它只影响姿态(pose), 不影响绑定(rest), 会让整套动作姿态塌缩到 ~1/100,
# 导致旋转正确但平移塌陷、下半身贴地。这里清除其 scale 动画并复位为 1。
if F.animation_data and F.animation_data.action:
    for fc in list(F.animation_data.action.fcurves):
        if fc.data_path == 'pose.bones["全ての親"].scale':
            F.animation_data.action.fcurves.remove(fc)
_rb=F.pose.bones.get("全ての親")
if _rb:
    _rb.scale=(1.0,1.0,1.0)
    print("reset 全ての親 pose scale -> (1,1,1)")
bpy.context.view_layer.update()

def hwr(a,n):
    bn=a.data.bones.get(n); return a.matrix_world@bn.head_local if bn else None
k=(hwr(M,"足.L")-hwr(M,"足首.L")).length/(hwr(F,"足.L")-hwr(F,"足首.L")).length
F.matrix_world=Matrix.Diagonal((k,k,k,1.0))@F.matrix_world; bpy.context.view_layer.update()
t=hwr(M,"足首.L")-hwr(F,"足首.L"); F.matrix_world=Matrix.Translation(t)@F.matrix_world; bpy.context.view_layer.update()

fs,fe=(int(x) for x in F.animation_data.action.frame_range)
bpy.context.scene.frame_start,bpy.context.scene.frame_end=fs,fe

# 接地: SMPL 源与银狼绑定姿态的腿长/地面参考不同, 整段动作可能整体悬空(或下沉)。
# 预扫一遍, 把动画中最低的脚踝对齐到 rest 脚踝高度(= MMD 地面), 再把 F 整体竖直平移。
# 这样 センター 与 足ＩＫ 一起下移, 双脚自然踩地, 不会贴地/悬空。
rest_ankle_z=min((hwr(M,"足首.L")).z,(hwr(M,"足首.R")).z)
min_foot=1e9
for _fr in range(fs,fe+1):
    bpy.context.scene.frame_set(_fr); bpy.context.view_layer.update()
    for _s in ("足首.L","足首.R"):
        _pb=F.pose.bones.get(_s)
        if _pb: min_foot=min(min_foot,(F.matrix_world@_pb.head).z)
ground_dz=min_foot-rest_ankle_z
F.matrix_world=Matrix.Translation((0,0,-ground_dz))@F.matrix_world; bpy.context.view_layer.update()
print("ground shift dz = %.4f (min_foot=%.4f rest_ankle=%.4f)"%(ground_dz,min_foot,rest_ankle_z))

IK_NAMEJ={"左足ＩＫ","右足ＩＫ","左つま先ＩＫ","右つま先ＩＫ"}
fb=set(x.name for x in F.data.bones)

CORE={"全ての親","センター","グルーブ","腰","上半身","上半身2","首","頭",
      "左肩","左腕","左ひじ","左手首","右肩","右腕","右ひじ","右手首",
      "下半身","左足","左ひざ","左足首","右足","右ひざ","右足首",
      "左足ＩＫ","左つま先ＩＫ","右足ＩＫ","右つま先ＩＫ"}
FINGERS=set()
for s in ("左","右"):
    for f in ("親指","人指","中指","薬指","小指"):
        for i in ("０","１","２","３"):
            FINGERS.add(s+f+i)
WHITE=CORE|FINGERS

# 约束(仅白名单, 避免尾/翼依赖环):
#   FK 骨 -> Copy Rotation(仅朝向, 不引入位移, 符合 MMD 旋转专用语义)
#   足ＩＫ -> Copy Transforms 自 足首; つま先ＩＫ -> Copy Location 自 つま先
#   センター -> 不加约束(后处理写入根位移)
def add_con(pb, target_name, kind):
    if target_name not in fb:
        print("  [warn] F missing", target_name); return
    c=pb.constraints.new(kind); c.target=F; c.subtarget=target_name

for pb in M.pose.bones:
    nj=pb.mmd_bone.name_j
    if nj not in WHITE:
        continue
    if nj=="センター":
        continue
    if nj in IK_NAMEJ:
        side=".L" if pb.name.endswith(".L") else (".R" if pb.name.endswith(".R") else "")
        if "つま先" in nj:
            add_con(pb, "つま先"+side, "COPY_LOCATION")
        else:
            add_con(pb, "足首"+side, "COPY_TRANSFORMS")
    elif pb.name in fb:
        add_con(pb, pb.name, "COPY_ROTATION")

bpy.ops.object.select_all(action="DESELECT"); M.select_set(True); bpy.context.view_layer.objects.active=M
for pb in M.pose.bones: pb.rotation_mode="QUATERNION"
bpy.ops.nla.bake(frame_start=fs,frame_end=fe,only_selected=False,visual_keying=True,
                 clear_constraints=True,clear_parents=False,use_current_action=True,bake_types={"POSE"})
print("bake done")

# 每骨转换矩阵(同 mmd_tools 导入侧, 不 invert)
def conv_mat(bone):
    m=bone.matrix_local.to_3x3()
    m[1],m[2]=m[2].copy(),m[1].copy()
    return m.transposed()

bones_export=[pb for pb in M.pose.bones if pb.mmd_bone.name_j in WHITE]
meta={}
for pb in bones_export:
    cm=conv_mat(pb.bone)
    meta[pb.name]=(cm, cm.to_quaternion(), cm.inverted())
print("export bones:", len(bones_export))

# 根位移: 直接把 M 的 腰 头对齐到 F 的 腰 头世界位置(绝对目标),
# 避免「delta vs rest」因 F/M 仅在脚踝对齐、髋部存在常量高度差而整体下沉。
center=M.pose.bones.get("センター")
center_rest_world=M.matrix_world@center.bone.matrix_local
IK_EXPORT={"左足ＩＫ","右足ＩＫ","左つま先ＩＫ","右つま先ＩＫ"}

# 默认插值: 取自参考 VMD, 否则线性默认
default_interp=None
if REF_VMD and os.path.isfile(REF_VMD):
    ref=vmd.File(); ref.load(filepath=REF_VMD)
    for keys in ref.boneAnimation.values():
        if keys: default_interp=list(keys[0].interp); break
if default_interp is None:
    default_interp=[20,20,0,0, 20,20,20,20, 107,107,107,107, 107,107,107,107]*4

out=vmd.File()
out.header=vmd.Header(); out.header.model_name="silver_wolf"
out.boneAnimation=vmd.BoneAnimation()

for fr in range(fs, fe+1):
    bpy.context.scene.frame_set(fr); bpy.context.view_layer.update()
    # センター 承载世界位移: 让 M 的 腰 头落到 F 的 腰 头世界位置(此刻 センター 处于 rest)
    d=(F.matrix_world@F.pose.bones["腰"].head)-(M.matrix_world@M.pose.bones["腰"].head)
    for pb in bones_export:
        nj=pb.mmd_bone.name_j
        cm,q,cm_inv=meta[pb.name]
        mb=pb.matrix_basis
        if nj=="センター":
            pb.matrix=Matrix.Translation(d)@center_rest_world
            bpy.context.view_layer.update()
            mb=pb.matrix_basis
            loc=mb.to_translation(); rot=Quaternion((1,0,0,0))
        elif nj in IK_EXPORT:
            loc=mb.to_translation(); rot=mb.to_quaternion()
        else:
            loc=Vector((0,0,0)); rot=mb.to_quaternion()
        vrot=(q.conjugated()@rot@q).normalized()      # (w,x,y,z)
        vloc=cm_inv@loc
        key=vmd.BoneFrameKey()
        key.frame_number=fr-fs
        key.location=(vloc.x, vloc.y, vloc.z)
        key.rotation=(vrot.x, vrot.y, vrot.z, vrot.w)
        key.interp=list(default_interp)
        out.boneAnimation[pb.mmd_bone.name_j].append(key)

out.save(filepath=OUT)
total=sum(len(v) for v in out.boneAnimation.values())
print("DONE custom VMD %s (%.1f KB, %d bones, %d keys)" % (
    OUT, os.path.getsize(OUT)/1024.0, len(out.boneAnimation), total))
