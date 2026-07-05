"""HY-Motion -> 银狼(silver_wolf) -> VMD 一体化编排器。

把仓库里原本分散、依赖不同运行环境（普通 Python / MotionBuilder mobupy / Blender）
的各个脚本，串成一条可一键运行（或分阶段运行）的流水线：

    [0] HY-Motion 文本生成     ->  .npz            (python，可选)
    [1] .npz/.npy -> SMPL FBX  ->  ASCII FBX       (python，run_reframe，任意帧数)
    [2] SMPL FBX  -> 银狼 FBX   ->  ASCII FBX       (mobupy，retarget.py)
    [3] 银狼 ASCII -> 二进制FBX ->  binary FBX      (mobupy，fbx_ascii_to_binary.py)
    [4] 银狼二进制 -> VMD       ->  .vmd            (blender，fbx_to_vmd_custom.py)

各阶段通过子进程调用既有脚本，互不污染环境。用 --stages 选择要跑的阶段。

用法示例（在仓库任意位置都可运行）:
    # 已有 .npz：放进 work_dir/00_npz，然后跑 1->4
    python run_pipeline.py --stages 1,2,3,4

    # 从文本开始（需先在 pipeline_config.yaml 里把 hymotion.enabled 设为 true）
    python run_pipeline.py --stages 0,1,2,3,4

    # 只跑某一阶段
    python run_pipeline.py --stages 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import yaml

# 目录结构:
#   <REPO_ROOT>/                         HY-Motion 仓库根
#     local_infer.py
#     smpl_silverwolf_pipeline/smpl_silverwolf_pipeline/   <- PKG_ROOT(本脚本所在)
#       run_pipeline.py
#       pipeline_config.yaml
#       fbx2npy2fbx/ retarget/ mmd/
PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PKG_ROOT, os.pardir, os.pardir))

FBX2NPY2FBX_DIR = os.path.join(PKG_ROOT, "fbx2npy2fbx")
RETARGET_DIR = os.path.join(PKG_ROOT, "retarget")
MMD_DIR = os.path.join(PKG_ROOT, "mmd")

ALL_STAGES = [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
#  辅助
# --------------------------------------------------------------------------- #
def resolve(path: Optional[str]) -> Optional[str]:
    """把配置里的路径解析为绝对路径（相对路径相对仓库根）。"""
    if path is None:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_files(directory: str, suffixes) -> List[str]:
    if not os.path.isdir(directory):
        return []
    out = [
        os.path.join(directory, n)
        for n in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, n)) and n.lower().endswith(suffixes)
    ]
    out.sort()
    return out


def run_cmd(cmd: List[str], cwd: Optional[str] = None, stage: str = "") -> None:
    """运行子进程；失败则抛异常并中断 pipeline。"""
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"\n[stage {stage}] $ {printable}")
    if cwd:
        print(f"[stage {stage}]   (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"[stage {stage}] 子进程失败（exit={result.returncode}）: {printable}")


def require_exe(path: Optional[str], name: str, stage: str) -> str:
    if not path:
        raise RuntimeError(f"[stage {stage}] 配置缺少可执行文件 '{name}'")
    # 允许 PATH 中的命令（如 "python"），不含路径分隔符时不强校验存在性
    if (os.sep in path or "/" in path) and not os.path.isfile(path):
        raise RuntimeError(f"[stage {stage}] 找不到 {name}: {path}")
    return path


# --------------------------------------------------------------------------- #
#  阶段实现
# --------------------------------------------------------------------------- #
def stage0_hymotion(cfg: Dict[str, Any], npz_dir: str) -> None:
    """[0] HY-Motion 文本 -> 动作 .npz。"""
    hm = cfg.get("hymotion", {}) or {}
    if not hm.get("enabled", False):
        print("[stage 0] hymotion.enabled=false，跳过文本生成阶段。")
        return

    python = require_exe(cfg["executables"].get("python", "python"), "python", "0")
    local_infer = os.path.join(REPO_ROOT, "local_infer.py")
    if not os.path.isfile(local_infer):
        raise RuntimeError(f"[stage 0] 找不到 local_infer.py: {local_infer}")

    model_path = resolve(hm.get("model_path"))
    input_text_dir = resolve(hm.get("input_text_dir"))

    cmd = [
        python, local_infer,
        "--model_path", model_path,
        "--output_dir", npz_dir,
        "--num_seeds", str(hm.get("num_seeds", 1)),
        "--cfg_scale", str(hm.get("cfg_scale", 5.0)),
    ]
    if input_text_dir:
        cmd += ["--input_text_dir", input_text_dir]
    if hm.get("device_ids"):
        cmd += ["--device_ids", str(hm["device_ids"])]
    if hm.get("disable_rewrite", True):
        cmd += ["--disable_rewrite"]
    if hm.get("disable_duration_est", True):
        cmd += ["--disable_duration_est"]
    if hm.get("validation_steps") is not None:
        cmd += ["--validation_steps", str(hm["validation_steps"])]

    # 以仓库根为 cwd，确保 examples / hymotion 包等相对路径可解析
    run_cmd(cmd, cwd=REPO_ROOT, stage="0")
    print(f"[stage 0] HY-Motion 动作已生成到: {npz_dir}")


def stage1_npy_to_fbx(cfg: Dict[str, Any], npz_dir: str, smpl_fbx_dir: str) -> None:
    """[1] .npz/.npy -> SMPL ASCII FBX（任意帧数）。"""
    python = require_exe(cfg["executables"].get("python", "python"), "python", "1")
    reframe = os.path.join(FBX2NPY2FBX_DIR, "npy_to_fbx", "run_reframe.py")
    if not os.path.isfile(reframe):
        raise RuntimeError(f"[stage 1] 找不到 run_reframe.py: {reframe}")

    template_fbx = resolve(cfg.get("npy_to_fbx", {}).get("template_fbx"))
    if not template_fbx or not os.path.isfile(template_fbx):
        raise RuntimeError(f"[stage 1] 找不到模板 FBX: {template_fbx}")

    motions = list_files(npz_dir, (".npz", ".npy"))
    if not motions:
        raise RuntimeError(f"[stage 1] 输入目录没有 .npz/.npy: {npz_dir}")
    print(f"[stage 1] 待转换 motion: {len(motions)} 个 (输入 {npz_dir})")

    cmd = [
        python, reframe,
        "--input", npz_dir,
        "--output", smpl_fbx_dir,
        "--template", template_fbx,
    ]
    if cfg.get("npy_to_fbx", {}).get("overwrite", True):
        cmd += ["--overwrite"]

    run_cmd(cmd, cwd=FBX2NPY2FBX_DIR, stage="1")
    print(f"[stage 1] SMPL FBX 输出: {smpl_fbx_dir}")


def stage2_retarget(cfg: Dict[str, Any], smpl_fbx_dir: str, retarget_dir: str) -> None:
    """[2] SMPL FBX -> 银狼 FBX（mobupy 重定向）。"""
    mobupy = require_exe(cfg["executables"].get("mobupy"), "mobupy", "2")
    retarget_py = os.path.join(RETARGET_DIR, "retarget.py")
    if not os.path.isfile(retarget_py):
        raise RuntimeError(f"[stage 2] 找不到 retarget.py: {retarget_py}")

    if not list_files(smpl_fbx_dir, (".fbx",)):
        raise RuntimeError(f"[stage 2] 输入目录没有 .fbx: {smpl_fbx_dir}")

    rt = cfg.get("retarget", {})
    cmd = [
        mobupy, retarget_py,
        "--source", rt.get("source", "SMPLX-lh-neutral"),
        "--target", rt.get("target", "silver_wolf"),
        "--input", smpl_fbx_dir,
        "--output", retarget_dir,
    ]
    run_cmd(cmd, cwd=RETARGET_DIR, stage="2")
    print(f"[stage 2] 银狼 FBX 输出: {retarget_dir}")


def stage3_ascii_to_binary(cfg: Dict[str, Any], retarget_dir: str, binary_dir: str) -> None:
    """[3] 银狼 ASCII FBX -> 二进制 FBX（mobupy，逐个文件）。"""
    mobupy = require_exe(cfg["executables"].get("mobupy"), "mobupy", "3")
    conv = os.path.join(MMD_DIR, "fbx_ascii_to_binary.py")
    if not os.path.isfile(conv):
        raise RuntimeError(f"[stage 3] 找不到 fbx_ascii_to_binary.py: {conv}")

    fbx_files = list_files(retarget_dir, (".fbx",))
    if not fbx_files:
        raise RuntimeError(f"[stage 3] 输入目录没有 .fbx: {retarget_dir}")
    os.makedirs(binary_dir, exist_ok=True)
    print(f"[stage 3] 待转二进制 FBX: {len(fbx_files)} 个")

    for i, src in enumerate(fbx_files, 1):
        dst = os.path.join(binary_dir, os.path.basename(src))
        print(f"[stage 3] [{i}/{len(fbx_files)}] {os.path.basename(src)}")
        run_cmd([mobupy, conv, src, dst], cwd=MMD_DIR, stage="3")
    print(f"[stage 3] 二进制 FBX 输出: {binary_dir}")


def stage4_fbx_to_vmd(cfg: Dict[str, Any], binary_dir: str, vmd_dir: str) -> None:
    """[4] 银狼二进制 FBX -> VMD（Blender + mmd_tools，逐个文件）。"""
    blender = require_exe(cfg["executables"].get("blender"), "blender", "4")
    script = os.path.join(MMD_DIR, "fbx_to_vmd_custom.py")
    if not os.path.isfile(script):
        raise RuntimeError(f"[stage 4] 找不到 fbx_to_vmd_custom.py: {script}")

    pmx = resolve(cfg.get("vmd", {}).get("pmx_model"))
    if not pmx or not os.path.isfile(pmx):
        raise RuntimeError(f"[stage 4] 找不到 PMX 模型: {pmx}")
    ref_vmd = resolve(cfg.get("vmd", {}).get("ref_vmd"))

    fbx_files = list_files(binary_dir, (".fbx",))
    if not fbx_files:
        raise RuntimeError(f"[stage 4] 输入目录没有 .fbx: {binary_dir}")
    os.makedirs(vmd_dir, exist_ok=True)
    print(f"[stage 4] 待转 VMD: {len(fbx_files)} 个")

    for i, src in enumerate(fbx_files, 1):
        out_vmd = os.path.join(vmd_dir, os.path.splitext(os.path.basename(src))[0] + ".vmd")
        print(f"[stage 4] [{i}/{len(fbx_files)}] {os.path.basename(src)} -> {os.path.basename(out_vmd)}")
        cmd = [blender, "-b", "--python", script, "--", pmx, src, out_vmd]
        if ref_vmd:
            cmd.append(ref_vmd)
        run_cmd(cmd, cwd=MMD_DIR, stage="4")
    print(f"[stage 4] VMD 输出: {vmd_dir}")
    print(f"[stage 4] 在 MMD 里加载 {pmx} + 上述 .vmd 即可播放银狼动作。")


# --------------------------------------------------------------------------- #
#  入口
# --------------------------------------------------------------------------- #
def parse_stages(value: str) -> List[int]:
    if value.strip().lower() == "all":
        return ALL_STAGES
    stages: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        s = int(part)
        if s not in ALL_STAGES:
            raise argparse.ArgumentTypeError(f"未知阶段 {s}，可选 {ALL_STAGES} 或 all")
        stages.append(s)
    return sorted(set(stages))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HY-Motion -> 银狼 -> VMD 一体化 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PKG_ROOT, "pipeline_config.yaml"),
        help="pipeline 配置 yaml 路径",
    )
    parser.add_argument(
        "--stages",
        type=parse_stages,
        default="all",
        help="要执行的阶段，逗号分隔（如 1,2,3,4），或 all。默认 all",
    )
    args = parser.parse_args()

    if isinstance(args.stages, str):  # default 字符串未经过 type 转换
        args.stages = parse_stages(args.stages)

    cfg = load_config(args.config)
    cfg.setdefault("executables", {})

    work_dir = resolve(cfg.get("work_dir", "output_pipeline"))
    dirs = cfg.get("dirs", {})
    npz_dir = os.path.join(work_dir, dirs.get("npz", "00_npz"))
    smpl_fbx_dir = os.path.join(work_dir, dirs.get("smpl_fbx", "01_smpl_fbx"))
    retarget_dir = os.path.join(work_dir, dirs.get("retarget_fbx", "02_silverwolf_fbx"))
    binary_dir = os.path.join(work_dir, dirs.get("binary_fbx", "03_silverwolf_bin"))
    vmd_dir = os.path.join(work_dir, dirs.get("vmd", "04_vmd"))

    for d in (npz_dir, smpl_fbx_dir, retarget_dir, binary_dir, vmd_dir):
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print("HY-Motion -> silver_wolf -> VMD pipeline")
    print(f"  config    : {args.config}")
    print(f"  repo root : {REPO_ROOT}")
    print(f"  work dir  : {work_dir}")
    print(f"  stages    : {args.stages}")
    print("=" * 70)

    dispatch = {
        0: lambda: stage0_hymotion(cfg, npz_dir),
        1: lambda: stage1_npy_to_fbx(cfg, npz_dir, smpl_fbx_dir),
        2: lambda: stage2_retarget(cfg, smpl_fbx_dir, retarget_dir),
        3: lambda: stage3_ascii_to_binary(cfg, retarget_dir, binary_dir),
        4: lambda: stage4_fbx_to_vmd(cfg, binary_dir, vmd_dir),
    }

    for s in args.stages:
        dispatch[s]()

    print("\n" + "=" * 70)
    print("pipeline 完成。")
    print(f"  SMPL FBX : {smpl_fbx_dir}")
    print(f"  银狼 FBX : {retarget_dir}")
    print(f"  二进制FBX: {binary_dir}")
    print(f"  VMD      : {vmd_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
