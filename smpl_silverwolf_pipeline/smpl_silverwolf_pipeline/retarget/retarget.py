"""SMPL <-> 银狼(silver_wolf) 互相重定向的通用驱动。

复用 mobu_retareting.run_pass，可指定任意 source/target（二者都需在
configs/characters_cfg.json 的 Characters 里、且 data/templates 下有对应 std.fbx）。

用法（用 MotionBuilder 的 mobupy 运行）：
    # SMPL -> 银狼
    mobupy.exe retarget.py --source SMPLX-lh-neutral --target silver_wolf \
        --input  <源动作FBX 或 目录> --output <输出目录>

    # 银狼 -> SMPL（反向）
    mobupy.exe retarget.py --source silver_wolf --target SMPLX-lh-neutral \
        --input  <源动作FBX 或 目录> --output <输出目录>

--input 可以是单个 .fbx，也可以是包含 .fbx 的目录（会递归收集）。
"""
import os
import sys
import json
from glob import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mobu_retareting as M


def parse_args(argv):
    opts = {"source": "SMPLX-lh-neutral", "target": "silver_wolf",
            "input": None, "output": os.path.join(ROOT, "output")}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--source", "--target", "--input", "--output") and i + 1 < len(argv):
            opts[a[2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    return opts


def collect_inputs(path):
    if path is None:
        return []
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return sorted(glob(os.path.join(path, "**", "*.fbx"), recursive=True))
    return [path] if os.path.isfile(path) else []


def main():
    opts = parse_args(sys.argv[1:])
    config_path = os.path.join(ROOT, "configs", "characters_cfg.json")
    with open(config_path, "r", encoding="utf-8") as f:
        character_cfg = json.load(f)
    skel_templates_root = character_cfg["asset_path"]["skel_templates_root"]

    input_files = collect_inputs(opts["input"])
    output_dir = os.path.abspath(opts["output"])

    print("source : {}".format(opts["source"]))
    print("target : {}".format(opts["target"]))
    print("inputs : {} 个".format(len(input_files)))
    print("output : {}".format(output_dir))
    if not input_files:
        print("没有可用的输入 FBX，退出。用 --input 指定文件或目录。")
        return

    outputs = M.run_pass(
        source_key=opts["source"],
        target_key=opts["target"],
        input_files=input_files,
        output_dir=output_dir,
        character_cfg=character_cfg,
        skel_templates_root=skel_templates_root,
    )
    print("\n完成，输出 {} 个文件:".format(len(outputs)))
    for o in outputs:
        print("  - {}".format(o))


if __name__ == "__main__":
    main()
