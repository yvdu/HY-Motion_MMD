"""把一批 SMPLX motion .npz/.npy 写回 ASCII FBX（支持任意帧数）。

与 npy_to_fbx/run.py 的区别：
    原 run.py 调用的 ascii_fbx_smplx_io.npy_to_ascii_fbx 要求 **模板帧数与 npy 帧数完全一致**，
    且只改写 KeyValueFloat 而不动 KeyTime，因此一个固定帧数的模板无法适配不同长度的动作。

本脚本复用 fbx2npy2fbx 里同一套解析逻辑，但会对模板里的每条 AnimationCurve 重建：
    * KeyTime       -> 按目标帧数 + fps 重新生成时间轴
    * KeyValueFloat -> 映射到的关节用 npz 数据，其余曲线按比例重采样
    * KeyAttrRefCount -> 同步成新的关键帧数
并更新 Take 的时间区间，从而得到帧数正确、可被 MotionBuilder 直接读取的 ASCII FBX。

输出的骨架/网格结构与现有可用源动作 alert.fbx / amzsprhnd.fbx 完全一致
（同一套 SMPLX-lh-neutral 骨架），可直接作为重定向源使用。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ascii_fbx_smplx_io import (  # noqa: E402
    _assemble_motion,
    _axis_angle_to_matrix,
    _format_array,
    _iter_blocks,
    _matrix_to_euler_xyz_deg,
)
from fbx_to_smplx_npz import FBX_KTIME_PER_SEC, SMPLX_JOINT_NAMES  # noqa: E402


def _load_motion(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """读取 .npz / .npy，返回 (poses[N,52,3], trans[N,3], fps)。"""
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=True) as data:
            poses = np.asarray(data["poses"], dtype=np.float64)
            trans = (
                np.asarray(data["trans"], dtype=np.float64)
                if "trans" in data.files
                else np.zeros((poses.shape[0], 3), dtype=np.float64)
            )
            fps = float(data["mocap_framerate"]) if "mocap_framerate" in data.files else 30.0
    else:
        data = np.load(path, allow_pickle=True).item()
        poses = np.asarray(data["poses"], dtype=np.float64)
        trans = np.asarray(data.get("trans", np.zeros((poses.shape[0], 3))), dtype=np.float64)
        fps = float(data.get("mocap_framerate", 30.0))
    poses = poses.reshape(poses.shape[0], -1, 3)
    if poses.shape[1] < 52:
        pad = np.zeros((poses.shape[0], 52 - poses.shape[1], 3), dtype=np.float64)
        poses = np.concatenate([poses, pad], axis=1)
    else:
        poses = poses[:, :52, :]
    return poses, trans, fps


def _build_curve_block(uid: int, default: str, times: np.ndarray, values: np.ndarray) -> str:
    n = len(values)
    return (
        f'\tAnimationCurve: {uid}, "AnimCurve::", "" {{\n'
        f"\t\tDefault: {default}\n"
        f"\t\tKeyVer: 4009\n"
        f"\t\tKeyTime: *{n} {{\n\t\t\ta: {_format_array(times, integer=True)}\n\t\t}} \n"
        f"\t\tKeyValueFloat: *{n} {{\n\t\t\ta: {_format_array(values)}\n\t\t}} \n"
        f"\t\tKeyAttrFlags: *1 {{\n\t\t\ta: 264\n\t\t}} \n"
        f"\t\tKeyAttrDataFloat: *4 {{\n\t\t\ta: 0,0,218434821,0\n\t\t}} \n"
        f"\t\tKeyAttrRefCount: *1 {{\n\t\t\ta: {n}\n\t\t}} \n"
        f"\t}}"
    )


def _resample(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == n:
        return values
    if len(values) <= 1:
        return np.full(n, values[0] if len(values) else 0.0, dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(values))
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, values)


def reframe_npy_to_fbx(template_fbx: str, motion_path: str, output_fbx: str) -> Tuple[int, float]:
    with open(template_fbx, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    payload, ctx = _assemble_motion(text, verbose=False)
    poses, trans, fps = _load_motion(motion_path)
    n_frames = poses.shape[0]
    eulers = _matrix_to_euler_xyz_deg(_axis_angle_to_matrix(poses))  # [N,52,3] deg

    curves = ctx["curves"]
    curve_to_animnode = ctx["curve_to_animnode"]
    animnode_to_model = ctx["animnode_to_model"]
    name_to_model = ctx["name_to_model"]
    joint_mapping = ctx["joint_mapping"]

    model_uid_to_joint: Dict[int, int] = {
        name_to_model[fbx_name]["uid"]: ji
        for ji, joint in enumerate(SMPLX_JOINT_NAMES)
        for fbx_name in [joint_mapping.get(joint)]
        if fbx_name and fbx_name in name_to_model
    }
    pelvis_uid = name_to_model[joint_mapping["pelvis"]]["uid"] if "pelvis" in joint_mapping else None

    kpf = int(round(FBX_KTIME_PER_SEC / fps))
    times = np.array([i * kpf for i in range(n_frames)], dtype=np.int64)

    # 预解析每条 AnimationCurve 的 Default 值
    default_re = re.compile(r"Default:\s*([-+0-9.eE]+)")

    chunks: List[str] = []
    cursor = 0
    n_replaced = 0
    n_resampled = 0
    for block in sorted(_iter_blocks(text, "AnimationCurve"), key=lambda b: b.start):
        chunks.append(text[cursor:block.start])
        cursor = block.end

        if block.uid not in curves:
            # 没有关键帧的曲线，原样保留
            chunks.append(text[block.start:block.end])
            continue

        m = default_re.search(block.body)
        default = m.group(1) if m else "0"

        values = None
        link = curve_to_animnode.get(block.uid)
        if link is not None:
            anuid, axis_name = link
            target = animnode_to_model.get(anuid)
            if target is not None:
                muid, prop = target
                axis_index = {"d|X": 0, "d|Y": 1, "d|Z": 2}.get(axis_name)
                if axis_index is not None:
                    if prop in ("Lcl Rotation", "LclRotation") and muid in model_uid_to_joint:
                        values = eulers[:, model_uid_to_joint[muid], axis_index]
                        n_replaced += 1
                    elif prop in ("Lcl Translation", "LclTranslation") and muid == pelvis_uid:
                        values = trans[:, axis_index] * 100.0
                        n_replaced += 1

        if values is None:
            values = _resample(curves[block.uid]["values"], n_frames)
            n_resampled += 1

        chunks.append(_build_curve_block(block.uid, default, times, np.asarray(values, dtype=np.float64)))

    chunks.append(text[cursor:])
    out_text = "".join(chunks)

    # 更新 Take 时间区间到新的帧数
    end_ktime = (n_frames - 1) * kpf
    out_text = re.sub(r"(LocalTime:\s*0,)\d+", rf"\g<1>{end_ktime}", out_text)
    out_text = re.sub(r"(ReferenceTime:\s*0,)\d+", rf"\g<1>{end_ktime}", out_text)

    out_dir = os.path.dirname(os.path.abspath(output_fbx))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_fbx, "w", encoding="utf-8", newline="\n") as f:
        f.write(out_text)
    return n_frames, fps


def list_motions(input_dir: str) -> List[str]:
    out: List[str] = []
    for name in os.listdir(input_dir):
        p = os.path.join(input_dir, name)
        if os.path.isfile(p) and name.lower().endswith((".npz", ".npy")):
            out.append(p)
    out.sort()
    return out


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_template = os.path.join(ROOT_DIR, "templates", "body_check_001__A296.fbx")
    parser = argparse.ArgumentParser(description="批量把 SMPLX .npz/.npy 写回 ASCII FBX(支持任意帧数)")
    parser.add_argument("--input", default="F:/mobu_retarget/output_npy_0622", help="输入 .npz/.npy 目录")
    parser.add_argument("--output", default="F:/mobu_retarget/output_fbx_0622", help="输出 .fbx 目录")
    parser.add_argument("--template", default=default_template, help="模板 ASCII FBX")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出")
    args = parser.parse_args()

    motions = list_motions(args.input)
    print(f"[扫描] 输入: {args.input}")
    print(f"[扫描] 找到 motion: {len(motions)} 个")
    print(f"[模板] {args.template}")
    print(f"[输出] {args.output}")
    os.makedirs(args.output, exist_ok=True)

    ok = 0
    fail = 0
    for i, motion in enumerate(motions, 1):
        out_path = os.path.join(args.output, os.path.splitext(os.path.basename(motion))[0] + ".fbx")
        if (not args.overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[{i}/{len(motions)}] 跳过(已存在): {out_path}")
            ok += 1
            continue
        try:
            n_frames, fps = reframe_npy_to_fbx(args.template, motion, out_path)
            print(f"[{i}/{len(motions)}] 完成 frames={n_frames} fps={fps:g}: {out_path}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"[{i}/{len(motions)}] 失败: {motion}\n{traceback.format_exc()}")
            fail += 1

    print(f"[完成] 成功={ok}, 失败={fail}")


if __name__ == "__main__":
    main()
