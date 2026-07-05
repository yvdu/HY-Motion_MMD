"""为 silver_wolf (MMD 银狼) 生成 HIK 角色化的 std.fbx 模板。

流程:
  1. mobupy 打开 silver_wolf_lv999.fbx
  2. 按 MMD 标准骨骼名 -> HIK Slot 建立映射, 创建 FBCharacter 并角色化
  3. 另存为 data/templates/silver_wolf/std.fbx (供 mobu_retareting.run_pass 作为 target 模板)

用法:
  mobupy.exe make_silver_wolf_template.py <input.fbx> <output_std.fbx>
"""
import os
import sys

import pyfbsdk as _pyfbsdk
if not hasattr(_pyfbsdk, "FBApplication"):
    import pyfbstandalone
    pyfbstandalone.initialize()
from pyfbsdk import *


CHARACTER_NAME = "silver_wolf"

# HIK Slot -> MMD 骨骼名 (mmd_tools 用 .L/.R 后缀; 手指用全角数字)
MAPPING = {
    "Reference": "全ての親",
    "Hips": "腰",
    "Spine": "上半身",
    "Spine1": "上半身1",
    "Spine2": "上半身2",
    "Neck": "首",
    "Head": "頭",

    "LeftShoulder": "肩.L",
    "LeftArm": "腕.L",
    "LeftForeArm": "ひじ.L",
    "LeftHand": "手首.L",
    "RightShoulder": "肩.R",
    "RightArm": "腕.R",
    "RightForeArm": "ひじ.R",
    "RightHand": "手首.R",

    "LeftUpLeg": "足.L",
    "LeftLeg": "ひざ.L",
    "LeftFoot": "足首.L",
    "LeftToeBase": "つま先.L",
    "RightUpLeg": "足.R",
    "RightLeg": "ひざ.R",
    "RightFoot": "足首.R",
    "RightToeBase": "つま先.R",

    # 手指 (全角数字)
    "LeftHandThumb1": "親指０.L", "LeftHandThumb2": "親指１.L", "LeftHandThumb3": "親指２.L",
    "LeftHandIndex1": "人指１.L", "LeftHandIndex2": "人指２.L", "LeftHandIndex3": "人指３.L",
    "LeftHandMiddle1": "中指１.L", "LeftHandMiddle2": "中指２.L", "LeftHandMiddle3": "中指３.L",
    "LeftHandRing1": "薬指１.L", "LeftHandRing2": "薬指２.L", "LeftHandRing3": "薬指３.L",
    "LeftHandPinky1": "小指１.L", "LeftHandPinky2": "小指２.L", "LeftHandPinky3": "小指３.L",

    "RightHandThumb1": "親指０.R", "RightHandThumb2": "親指１.R", "RightHandThumb3": "親指２.R",
    "RightHandIndex1": "人指１.R", "RightHandIndex2": "人指２.R", "RightHandIndex3": "人指３.R",
    "RightHandMiddle1": "中指１.R", "RightHandMiddle2": "中指２.R", "RightHandMiddle3": "中指３.R",
    "RightHandRing1": "薬指１.R", "RightHandRing2": "薬指２.R", "RightHandRing3": "薬指３.R",
    "RightHandPinky1": "小指１.R", "RightHandPinky2": "小指２.R", "RightHandPinky3": "小指３.R",
}

# 角色化最低要求的 Slot (缺一不可)
REQUIRED = [
    "Hips", "Spine", "Head",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
    "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
]


def find_model_by_name(name):
    def _find(model):
        if model.Name == name:
            return model
        for child in model.Children:
            r = _find(child)
            if r:
                return r
        return None
    for model in FBSystem().Scene.RootModel.Children:
        r = _find(model)
        if r:
            return r
    return None


def main():
    in_fbx = os.path.abspath(sys.argv[1])
    out_fbx = os.path.abspath(sys.argv[2])

    app = FBApplication()
    app.FileNew()
    print("打开:", in_fbx)
    app.FileOpen(in_fbx)

    character = FBCharacter(CHARACTER_NAME)

    mapped = []
    missing = []
    for slot, bone in MAPPING.items():
        model = find_model_by_name(bone)
        if not model:
            missing.append((slot, bone))
            continue
        prop = character.PropertyList.Find(slot + "Link")
        if prop is None:
            missing.append((slot, bone + " (no slot)"))
            continue
        prop.append(model)
        mapped.append(slot)

    print("已映射 {} 个 slot:".format(len(mapped)))
    print("  " + ", ".join(mapped))
    if missing:
        print("未映射/未找到:")
        for slot, bone in missing:
            print("  {} <- {}".format(slot, bone))

    miss_required = [s for s in REQUIRED if s not in mapped]
    if miss_required:
        print("错误: 必需 slot 缺失: {}".format(miss_required))
        sys.exit(3)

    ok = character.SetCharacterizeOn(True)
    print("SetCharacterizeOn ->", ok)
    if not ok:
        print("错误: 角色化失败")
        sys.exit(4)

    FBApplication().CurrentCharacter = character

    out_dir = os.path.dirname(out_fbx)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    print("保存模板:", out_fbx)
    saved = app.FileSave(out_fbx)
    print("FileSave ->", saved)
    if os.path.isfile(out_fbx):
        size_mb = os.path.getsize(out_fbx) / (1024.0 * 1024.0)
        print("DONE  {}  ({:.2f} MB)".format(out_fbx, size_mb))
    else:
        print("错误: 未生成模板文件")
        sys.exit(2)


if __name__ == "__main__":
    main()
