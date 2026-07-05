"""把单个 SMPLX 动作重定向到 silver_wolf (银狼)。

    mobupy.exe retarget_silver_wolf.py [输入FBX] [输出目录]
默认输入: output_0622_fbx/00000002_000.fbx
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mobu_retareting as M

DEFAULT_INPUT = os.path.join(ROOT, "output_0622_fbx", "00000002_000.fbx")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "output_retarget_silver_wolf")


def main():
    input_file = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR

    config_path = os.path.join(ROOT, "configs", "characters_cfg.json")
    with open(config_path, "r", encoding="utf-8") as f:
        character_cfg = json.load(f)
    skel_templates_root = character_cfg["asset_path"]["skel_templates_root"]

    print("源动作: {}".format(input_file))
    print("输出目录: {}".format(output_dir))
    if not os.path.isfile(input_file):
        print("找不到输入 FBX, 退出。")
        return

    print("\n########## SMPLX-lh-neutral -> silver_wolf (直接重定向) ##########")
    outputs = M.run_pass(
        source_key="SMPLX-lh-neutral",
        target_key="silver_wolf",
        input_files=[input_file],
        output_dir=output_dir,
        character_cfg=character_cfg,
        skel_templates_root=skel_templates_root,
    )

    print("\n完成, 输出 {} 个文件:".format(len(outputs)))
    for o in outputs:
        print("  - {}".format(o))


if __name__ == "__main__":
    main()
