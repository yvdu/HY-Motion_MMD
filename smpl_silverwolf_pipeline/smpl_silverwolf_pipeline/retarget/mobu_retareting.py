import os

import pyfbsdk as _pyfbsdk
if not hasattr(_pyfbsdk, "FBApplication"):
    import pyfbstandalone
    pyfbstandalone.initialize()
from pyfbsdk import *
try:
    from pyfbsdk_additions import *
except Exception:
    pass


def import_fbx(file_path):
    # NO_LOAD_UI_DIALOG = False
    app = FBApplication()
    app.FileOpen(file_path)
    print("Imported:", file_path)

def merge_fbx(file_path):
    app = FBApplication()
    app.FileMerge(file_path)
    print(f"FileMerged: {file_path}")

def find_take_by_name(take_name):
    take = None
    for t in FBSystem().Scene.Takes:
        if t.Name == take_name:
            take = t
            break
    if not take:
        print(f"未找到动画轨道: {take_name}")
        return
    FBSystem().CurrentTake = take
    
def find_character_by_name(character_name):
    """根据名称查找角色"""
    for char in FBSystem().Scene.Characters:
        if char.Name == character_name:
            return char
    return None

def find_model_by_name(name):
    """递归查找模型节点"""
    def _find(model, name):
        if model.Name == name:
            return model
        for child in model.Children:
            result = _find(child, name)
            if result:
                return result
        return None
    for model in FBSystem().Scene.RootModel.Children:
        result = _find(model, name)
        if result:
            return result
    return None

def find_model_by_longname(longname):
    """按 LongName(含命名空间, 如 'PC:Bip001') 精确查找模型节点。"""
    result = []
    def _walk(model):
        if model.LongName == longname:
            result.append(model)
        for child in model.Children:
            _walk(child)
    for model in FBSystem().Scene.RootModel.Children:
        _walk(model)
    return result[0] if result else None

def apply_namespace(namespace):
    """给当前场景中的所有角色与模型加上命名空间(隔离同名骨骼, 如两套 Bip001)。"""
    scene = FBSystem().Scene
    for c in list(scene.Characters):
        c.ProcessObjectNamespace(FBNamespaceAction.kFBConcatNamespace, namespace, None, True)
    for m in list(scene.RootModel.Children):
        m.ProcessObjectNamespace(FBNamespaceAction.kFBConcatNamespace, namespace, None, True)

def strip_namespace_in_ascii_fbx(fbx_path, namespace):
    """在导出的 ASCII FBX 文件中移除临时命名空间前缀。

    命名空间隔离导出时只选择 target 对象，因此全局移除 "TGT:" 是安全的；
    这样可以避免 MotionBuilder ProcessObjectNamespace 剥离 namespace 时销毁 SDK 对象引用。
    """
    prefix = f"{namespace}:"
    with open(fbx_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    if prefix in data:
        with open(fbx_path, "w", encoding="utf-8", newline="") as f:
            f.write(data.replace(prefix, ""))
        print(f"已移除导出FBX中的临时命名空间: {prefix}")

def get_skeleton_root(character, root_name):
    """从角色对象自身的骨骼层级中解析骨骼根节点。

    相比按名字全局查找(find_model_by_name)，这里从角色的 Hips 向上回溯，
    只在该角色自己的层级里寻找名为 root_name 的祖先。这样即使场景中存在
    多个同名根骨骼(例如两个角色都叫 Bip001)，也能拿到正确的那一个。
    """
    try:
        hips = character.GetModel(FBBodyNodeId.kFBHipsNodeId)
    except Exception:
        hips = None
    if not hips:
        return None
    node = hips
    top = hips
    while node:
        if root_name and node.Name == root_name:
            return node
        top = node
        node = node.Parent
    return top

def SelectBranch(topModel):
    for childModel in topModel.Children:
        SelectBranch(childModel)
    topModel.Selected = True

def export_select_to_fbx(root_model, root_meshes, export_fbx_path, character=None):
    """递归选择骨架和mesh, 导出FBX。

    若传入 character, 则同时选中该角色, 使导出的FBX保留角色(HIK)约束,
    便于结果作为后续重定向的源角色复用(与原始数据格式一致)。
    """
    # 先清空已有选择，避免误导出其它对象(如源角色)
    for comp in FBSystem().Scene.Components:
        comp.Selected = False

    SelectBranch(root_model)
    
    # 如果root_meshes是列表，选择所有mesh
    if isinstance(root_meshes, list):
        for mesh in root_meshes:
            if mesh:
                SelectBranch(mesh)
    else:
        # 如果是单个mesh对象
        SelectBranch(root_meshes)

    # 选中目标角色，使其HIK约束一并导出
    if character:
        character.Selected = True

    options = FBFbxOptions(False)
    options.UseASCIIFormat = True
    options.SaveSelectedModelsOnly = True
    # 可选：不导出摄像机、灯光等
    options.BaseCameras = False
    options.CameraSwitcherSettings = False
    options.CurrentCameraSettings = False
    options.GlobalLightingSettings = False
    options.TransportSettings = False

    FBApplication().FileSave(export_fbx_path, options)
    print(f"已导出FBX: {export_fbx_path}")

def bake_animation_to_character(character):
    """创建烘焙选项, 烘焙动画到角色骨架"""
    plot_options = FBPlotOptions()
    plot_options.PlotAllTakes = False
    plot_options.PlotOnFrame = True
    plot_options.PlotPeriod = FBTime(0,0,0,1)  # 每帧烘焙
    plot_options.PlotTranslationOnRootOnly = False
    plot_options.UseConstantKeyReducer = False

    character.PlotAnimation(FBCharacterPlotWhere.kFBCharacterPlotOnSkeleton, plot_options)
    print(f"已烘焙动画到角色: {character.Name}")


def apply_root_rotation_fix(post_fix):
    """烘焙后修正根/Reference 节点的局部旋转(朝向后处理)。

    某些来源(如 MMD 经 Blender 转出的 FBX)在 mobu 内导入后, Reference 节点
    (如 '全ての親')会带一个固定旋转偏移, 导致重定向结果整体朝向错误(如面朝地面)。
    通过 config 里的 post_fix 指定该节点与目标旋转, 在烘焙后、导出前强制覆盖:
        "post_fix": {"bone": "全ての親", "rotation": [180, 0, 0]}
    会清除该节点已有的旋转动画关键帧, 再写入静态旋转(该节点本就不承载实际动作)。
    """
    if not post_fix:
        return
    bone = post_fix.get("bone")
    rot = post_fix.get("rotation")
    if not bone or rot is None:
        return
    model = find_model_by_name(bone)
    if not model:
        print(f"post_fix: 未找到节点 {bone}, 跳过朝向修正")
        return
    # 清除该节点旋转动画(否则烘焙出的关键帧会覆盖静态值)
    anim_node = model.Rotation.GetAnimationNode()
    if anim_node:
        for sub in anim_node.Nodes:
            if sub.FCurve:
                sub.FCurve.EditClear()
    model.Rotation = FBVector3d(float(rot[0]), float(rot[1]), float(rot[2]))
    print(f"post_fix: 已将 {bone} 局部旋转设为 {rot}")


def retargeting(animation_path, character_source, character_target):
    """合并动画FBX"""
    skeleton_c_objects = merge_fbx(animation_path)

    """设置Character Controls面板"""
    # Use find_character_by_name to ensure we get FBCharacter objects
    Character = find_character_by_name(character_target)
    if not Character:
        print(f"Error: Target character '{character_target}' not found!")
        return
    Character.Selected = True

    OldCharacter = find_character_by_name(character_source)
    if not OldCharacter:
        print(f"Error: Source character '{character_source}' not found!")
        return
    
    print(f"Source character: {OldCharacter.Name}, Target character: {Character.Name}")
    Character.InputCharacter = OldCharacter
    Character.InputType = FBCharacterInputType.kFBCharacterInputCharacter
    Character.ActiveInput = True

def bake_and_export(export_path, 
                    character_name="SMPLX",
                    character_model_name="SMPLX-neutral", 
                    character_mesh_names=["SMPLX-mesh-neutral"], 
                    take_name="Take 001",
                    post_fix=None):
    
    # 切换至Take 001
    find_take_by_name(take_name)

    # 将动画烘焙到骨骼character_name
    character = find_character_by_name(character_name)
    if not character:
        print(f"Error: character '{character_name}' not found! 跳过导出: {export_path}")
        return

    # 优先从角色对象自身层级解析骨骼根，避免同名根骨骼(如 Bip001)歧义
    character_model = get_skeleton_root(character, character_model_name)
    if not character_model:
        character_model = find_model_by_name(character_model_name)
    
    # 查找所有mesh模型
    character_meshes = []
    for mesh_name in character_mesh_names:
        mesh = find_model_by_name(mesh_name)
        if mesh:
            character_meshes.append(mesh)
            print(f"找到mesh: {mesh.Name}")
        else:
            print(f"警告: 未找到mesh: {mesh_name}")
    
    print(f"找到角色: {character.Name}, 根模型: {character_model.Name}")
    bake_animation_to_character(character)

    # 烘焙后朝向后处理(可选)
    apply_root_rotation_fix(post_fix)

    # 导出选择的骨架和网格(同时保留目标角色HIK约束)
    export_select_to_fbx(character_model, character_meshes, export_path, character)

def bake_and_export_isolated(export_path, namespace,
                             target_char, target_root, target_mesh_names,
                             source_char, source_root, source_mesh_names,
                             take_name="Take 001", post_fix=None):
    """命名空间隔离场景下的烘焙+导出(用于 source/target 根骨骼同名的情况)。

    target 已被加上 namespace(如 'PC'), source 无命名空间。流程:
      1. 烘焙到 target 骨骼
      2. 删除 source 角色/骨骼/网格
      3. 去掉 target 命名空间 -> 导出干净节点名
    """
    find_take_by_name(take_name)

    character = find_character_by_name(target_char)
    if not character:
        print(f"Error: target character '{target_char}' not found! 跳过导出: {export_path}")
        return
    bake_animation_to_character(character)

    # 在删除 source/剥离 namespace 之前先保存 target 对象引用。
    # 剥离 namespace 后按名字重新查找可能因为同名对象清理顺序失败，但对象引用仍然有效。
    character_model = find_model_by_longname(f"{namespace}:{target_root}")
    if not character_model:
        character_model = find_model_by_name(target_root)

    target_meshes = []
    mesh_names = target_mesh_names if isinstance(target_mesh_names, list) else [target_mesh_names]
    for mesh_name in mesh_names:
        mesh = find_model_by_longname(f"{namespace}:{mesh_name}") or find_model_by_name(mesh_name)
        if mesh:
            target_meshes.append(mesh)
            print(f"找到target mesh: {mesh.LongName}")
        elif mesh_name:
            print(f"警告: 未找到target mesh: {mesh_name}")

    # 烘焙后朝向后处理(可选)
    apply_root_rotation_fix(post_fix)

    target_character = find_character_by_name(target_char)
    print(f"导出角色: {target_char}, 根骨骼: {character_model.LongName if character_model else None}")
    export_select_to_fbx(character_model, target_meshes, export_path, target_character)
    strip_namespace_in_ascii_fbx(export_path, namespace)

def delete_all_animation_on_current_take():
    current_take = FBSystem().CurrentTake
    current_take.ClearAllPropertiesOnCurrentLayer()
    print(f"已删除当前Take（{current_take.Name}）上的所有动画")


def run_pass(source_key, target_key, input_files, output_dir, character_cfg, skel_templates_root):
    """执行单步重定向: source_key -> target_key。

    对 input_files 中的每个动画FBX:
      1. 打开 source T-Pose 模板(FileOpen, 重置场景)
      2. 合并 target T-Pose 模板(FileMerge)
      3. 合并动画并设置 Source->Target 重定向
      4. 烘焙到 target 骨骼并导出
    返回导出的FBX路径列表(可作为下一步重定向的输入, 用于链式重定向)。
    """
    source_char = character_cfg["Characters"][source_key]["character_name"]
    target_char = character_cfg["Characters"][target_key]["character_name"]
    source_root = character_cfg["Characters"][source_key]["root_skeleton"]
    target_root = character_cfg["Characters"][target_key]["root_skeleton"]
    target_mesh = character_cfg["Characters"][target_key]["mesh_names"]
    source_mesh = character_cfg["Characters"][source_key]["mesh_names"]
    target_post_fix = character_cfg["Characters"][target_key].get("post_fix")

    # 允许 skel_templates_root 写成相对路径(相对本文件所在的包根目录),
    # 这样整个包可以拷贝到任意位置直接运行。
    if not os.path.isabs(skel_templates_root):
        skel_templates_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), skel_templates_root
        ).replace("\\", "/")

    template_source = f"{skel_templates_root}/{source_key}/std.fbx"
    template_target = f"{skel_templates_root}/{target_key}/std.fbx"

    # source 与 target 的根骨骼/角色同名时(如都用 Bip001),
    # 直接合并会因同名而串骨, 需要给 target 加命名空间隔离。
    need_namespace = (source_root == target_root) or (source_char == target_char)
    NS = "TGT"

    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for src_anim in input_files:
        print(f"\n=== [{source_key}({source_char}) -> {target_key}({target_char})] 处理动画: {src_anim} "
              f"(namespace隔离={need_namespace}) ===")

        export_path = os.path.join(output_dir, os.path.basename(src_anim))

        if need_namespace:
            # 先加载并隔离 target, 再加载 source 模板, 最后合并动画
            import_fbx(template_target)
            apply_namespace(NS)
            merge_fbx(template_source)
            retargeting(src_anim, source_char, target_char)
            bake_and_export_isolated(export_path, NS,
                                     target_char, target_root, target_mesh,
                                     source_char, source_root, source_mesh,
                                     post_fix=target_post_fix)
        else:
            # 标准流程: 加载 source & target T-Pose, 重定向, 烘焙导出
            import_fbx(template_source)
            merge_fbx(template_target)
            retargeting(src_anim, source_char, target_char)
            bake_and_export(export_path, target_char, target_root, target_mesh,
                            post_fix=target_post_fix)

        delete_all_animation_on_current_take()
        outputs.append(export_path)

    return outputs