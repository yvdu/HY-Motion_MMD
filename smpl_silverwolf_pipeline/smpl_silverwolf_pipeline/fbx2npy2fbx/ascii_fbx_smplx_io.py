"""
把 ASCII SMPLX FBX 转成项目约定的 motion .npy 格式，
也支持使用源 FBX 作为模板，把 .npy 中的动画曲线写回 FBX。

二进制 FBX 仍由 fbx_to_smplx_npz.py 处理；本模块保持同一套输出字段：
    poses / trans / betas / gender / mocap_framerate
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from fbx_to_smplx_npz import (
    FBX_KTIME_PER_SEC,
    SMPLX_JOINT_NAMES,
    build_joint_mapping,
    euler_xyz_deg_to_matrix,
    matrix_to_axis_angle,
    parse_fbx_object_name,
    sample_curve,
)


@dataclass
class Block:
    uid: int
    name: str
    kind: str
    start: int
    end: int
    body: str


def _find_matching_brace(text: str, open_brace: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for i in range(open_brace, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("ASCII FBX 中存在未匹配的大括号")


def _iter_blocks(text: str, block_name: str) -> Iterable[Block]:
    pattern = re.compile(
        rf"(?m)^[ \t]*{re.escape(block_name)}:\s*(-?\d+),\s*\"([^\"]*)\",\s*\"([^\"]*)\"\s*\{{"
    )
    for match in pattern.finditer(text):
        open_brace = text.find("{", match.start(), match.end())
        close_brace = _find_matching_brace(text, open_brace)
        yield Block(
            uid=int(match.group(1)),
            name=parse_fbx_object_name(match.group(2)),
            kind=parse_fbx_object_name(match.group(3)),
            start=match.start(),
            end=close_brace + 1,
            body=text[open_brace + 1 : close_brace],
        )


def _parse_property_vec3(body: str, prop_name: str, default: float = 0.0) -> np.ndarray:
    pattern = re.compile(
        rf'P:\s*"{re.escape(prop_name)}"\s*,[^\n]*?,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)'
    )
    match = pattern.search(body)
    if not match:
        return np.full(3, default, dtype=np.float64)
    return np.array([float(match.group(i)) for i in range(1, 4)], dtype=np.float64)


def _parse_property_scalar(body: str, prop_name: str, default: float = 0.0) -> float:
    pattern = re.compile(rf'P:\s*"{re.escape(prop_name)}"\s*,[^\n]*?,\s*([-+0-9.eE]+)')
    match = pattern.search(body)
    return float(match.group(1)) if match else default


def _parse_numeric_array(body: str, field_name: str, dtype: Any) -> np.ndarray:
    pattern = re.compile(rf"{re.escape(field_name)}:\s*\*\d+\s*\{{\s*a:\s*(.*?)\n[ \t]*\}}", re.S)
    match = pattern.search(body)
    if not match:
        return np.asarray([], dtype=dtype)
    payload = re.sub(r"\s+", "", match.group(1))
    return np.fromstring(payload, sep=",", dtype=dtype)


def _extract_models(text: str) -> Dict[int, Dict[str, Any]]:
    models: Dict[int, Dict[str, Any]] = {}
    for block in _iter_blocks(text, "Model"):
        models[block.uid] = {
            "uid": block.uid,
            "name": block.name,
            "type": block.kind,
            "lcl_trans": _parse_property_vec3(block.body, "Lcl Translation"),
            "lcl_rot": _parse_property_vec3(block.body, "Lcl Rotation"),
            "rotation_order": int(_parse_property_scalar(block.body, "RotationOrder", 0)),
        }
    return models


def _extract_anim_curves(text: str) -> Dict[int, Dict[str, Any]]:
    curves: Dict[int, Dict[str, Any]] = {}
    for block in _iter_blocks(text, "AnimationCurve"):
        times = _parse_numeric_array(block.body, "KeyTime", np.int64)
        values = _parse_numeric_array(block.body, "KeyValueFloat", np.float64)
        if len(times) == 0 or len(values) == 0:
            continue
        curves[block.uid] = {
            "uid": block.uid,
            "times": times,
            "values": values,
            "span": (block.start, block.end),
        }
    return curves


def _extract_anim_curve_nodes(text: str) -> Dict[int, Dict[str, Any]]:
    nodes: Dict[int, Dict[str, Any]] = {}
    for block in _iter_blocks(text, "AnimationCurveNode"):
        defaults = {
            "d|X": _parse_property_scalar(block.body, "d|X", 0.0),
            "d|Y": _parse_property_scalar(block.body, "d|Y", 0.0),
            "d|Z": _parse_property_scalar(block.body, "d|Z", 0.0),
        }
        nodes[block.uid] = {"uid": block.uid, "name": block.name, "defaults": defaults}
    return nodes


def _extract_connections(text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    conn_start = text.find("Connections:")
    if conn_start < 0:
        return []
    open_brace = text.find("{", conn_start)
    close_brace = _find_matching_brace(text, open_brace)
    body = text[open_brace + 1 : close_brace]
    pattern = re.compile(r'C:\s*"([^"]+)",\s*(-?\d+),\s*(-?\d+)(?:,\s*"([^"]+)")?')
    out: List[Tuple[str, int, int, Optional[str]]] = []
    for match in pattern.finditer(body):
        ctype = match.group(1)
        if ctype not in ("OO", "OP"):
            continue
        out.append((ctype, int(match.group(2)), int(match.group(3)), match.group(4)))
    return out


def _extract_fps(text: str, fps_override: Optional[float]) -> float:
    if fps_override:
        return float(fps_override)
    fps = 30.0
    time_mode = None
    custom_fps = None
    match = re.search(r'P:\s*"TimeMode"\s*,[^\n]*?,\s*(-?\d+)', text)
    if match:
        time_mode = int(match.group(1))
    match = re.search(r'P:\s*"CustomFrameRate"\s*,[^\n]*?,\s*([-+0-9.eE]+)', text)
    if match:
        custom_fps = float(match.group(1))
    time_mode_map = {
        0: 30.0, 1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0,
        5: 48.0, 6: 30.0, 7: 30.0, 8: 29.97, 9: 29.97,
        10: 25.0, 11: 24.0, 12: 23.976, 13: 24.0, 14: -1.0,
        15: 96.0, 16: 72.0, 17: 59.94,
    }
    if time_mode in time_mode_map and time_mode_map[time_mode] > 0:
        fps = time_mode_map[time_mode]
    if custom_fps and custom_fps > 0 and (time_mode == 14 or fps == 30.0):
        fps = custom_fps
    return fps


def _build_curve_links(
    models: Dict[int, Dict[str, Any]],
    curves: Dict[int, Dict[str, Any]],
    anim_nodes: Dict[int, Dict[str, Any]],
    connections: List[Tuple[str, int, int, Optional[str]]],
) -> Tuple[Dict[int, Tuple[int, str]], Dict[int, Tuple[int, str]]]:
    curve_to_animnode: Dict[int, Tuple[int, str]] = {}
    animnode_to_model: Dict[int, Tuple[int, str]] = {}
    for ctype, src, dst, prop in connections:
        if ctype != "OP" or not prop:
            continue
        if src in curves and dst in anim_nodes and prop in ("d|X", "d|Y", "d|Z"):
            curve_to_animnode[src] = (dst, prop)
        elif src in anim_nodes and dst in models:
            animnode_to_model[src] = (dst, prop)
    return curve_to_animnode, animnode_to_model


def _assemble_motion(
    text: str, fps_override: Optional[float] = None, verbose: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fps = _extract_fps(text, fps_override)
    models = _extract_models(text)
    curves = _extract_anim_curves(text)
    anim_nodes = _extract_anim_curve_nodes(text)
    connections = _extract_connections(text)
    curve_to_animnode, animnode_to_model = _build_curve_links(models, curves, anim_nodes, connections)

    name_to_model: Dict[str, Dict[str, Any]] = {}
    for model in models.values():
        if model["name"] not in name_to_model or model["type"] in ("LimbNode", "Limb"):
            name_to_model[model["name"]] = model

    model_curves: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for muid, model in models.items():
        model_curves[muid] = {
            "T": {"X": None, "Y": None, "Z": None, "defaults": model["lcl_trans"].tolist()},
            "R": {"X": None, "Y": None, "Z": None, "defaults": model["lcl_rot"].tolist()},
        }

    for cuid, (anuid, axis) in curve_to_animnode.items():
        if anuid not in animnode_to_model:
            continue
        muid, prop = animnode_to_model[anuid]
        if muid not in model_curves:
            continue
        ax = axis[-1]
        if prop in ("Lcl Translation", "LclTranslation"):
            model_curves[muid]["T"][ax] = curves[cuid]
        elif prop in ("Lcl Rotation", "LclRotation"):
            model_curves[muid]["R"][ax] = curves[cuid]
        an_default = anim_nodes[anuid]["defaults"].get(axis)
        if an_default is not None:
            ai = {"X": 0, "Y": 1, "Z": 2}[ax]
            if prop in ("Lcl Translation", "LclTranslation"):
                model_curves[muid]["T"]["defaults"][ai] = an_default
            elif prop in ("Lcl Rotation", "LclRotation"):
                model_curves[muid]["R"]["defaults"][ai] = an_default

    joint_mapping = build_joint_mapping(list(name_to_model.keys()))
    all_times: List[np.ndarray] = []
    for joint in SMPLX_JOINT_NAMES:
        fbx_name = joint_mapping.get(joint)
        if not fbx_name:
            continue
        muid = name_to_model[fbx_name]["uid"]
        for transform in ("T", "R"):
            for ax in ("X", "Y", "Z"):
                curve = model_curves[muid][transform][ax]
                if curve is not None and len(curve["times"]) > 0:
                    all_times.append(curve["times"])
    if not all_times:
        raise RuntimeError("ASCII FBX 中未找到动画曲线")

    t0 = min(int(times.min()) for times in all_times)
    t1 = max(int(times.max()) for times in all_times)
    duration_sec = (t1 - t0) / FBX_KTIME_PER_SEC
    num_frames = max(1, int(round(duration_sec * fps)) + 1)
    sample_times = np.linspace(t0, t1, num_frames).astype(np.int64)

    poses = np.zeros((num_frames, 52, 3), dtype=np.float64)
    trans = np.zeros((num_frames, 3), dtype=np.float64)
    for ji, joint in enumerate(SMPLX_JOINT_NAMES):
        fbx_name = joint_mapping.get(joint)
        if not fbx_name:
            continue
        muid = name_to_model[fbx_name]["uid"]
        rblock = model_curves[muid]["R"]
        eulers = np.zeros((num_frames, 3), dtype=np.float64)
        for ai, ax in enumerate(("X", "Y", "Z")):
            eulers[:, ai] = sample_curve(rblock[ax], rblock["defaults"][ai], sample_times)
        poses[:, ji, :] = matrix_to_axis_angle(euler_xyz_deg_to_matrix(eulers))

    if "pelvis" in joint_mapping:
        muid = name_to_model[joint_mapping["pelvis"]]["uid"]
        tblock = model_curves[muid]["T"]
        for ai, ax in enumerate(("X", "Y", "Z")):
            trans[:, ai] = sample_curve(tblock[ax], tblock["defaults"][ai], sample_times)
        trans /= 100.0

    if verbose:
        missing = [joint for joint in SMPLX_JOINT_NAMES if joint not in joint_mapping]
        print(f"[ASCII FBX] fps: {fps}")
        print(f"[Map] matched {len(joint_mapping)} / 52 SMPLX joints")
        if missing:
            print(f"[Map] missing: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        print(f"[Anim] frames: {num_frames}, duration: {duration_sec:.3f}s")

    payload = {
        "poses": poses.reshape(num_frames, -1).astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": np.zeros(16, dtype=np.float32),
        "gender": "neutral",
        "mocap_framerate": np.float32(fps),
    }
    context = {
        "fps": fps,
        "models": models,
        "curves": curves,
        "anim_nodes": anim_nodes,
        "connections": connections,
        "curve_to_animnode": curve_to_animnode,
        "animnode_to_model": animnode_to_model,
        "name_to_model": name_to_model,
        "joint_mapping": joint_mapping,
    }
    return payload, context


def ascii_fbx_to_npy(
    fbx_path: str, npy_path: str, fps_override: Optional[float] = None, verbose: bool = True
) -> None:
    with open(fbx_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    payload, _context = _assemble_motion(text, fps_override=fps_override, verbose=verbose)
    out_dir = os.path.dirname(os.path.abspath(npy_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(npy_path, np.array(payload, dtype=object), allow_pickle=True)
    if verbose:
        print(f"[Save] -> {npy_path}")
        print(f"        poses {payload['poses'].shape}, trans {payload['trans'].shape}, fps {payload['mocap_framerate']}")


def _axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec, axis=-1)
    axis = np.zeros_like(rotvec)
    safe = theta > 1e-12
    axis[safe] = rotvec[safe] / theta[safe, None]
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c
    R = np.zeros(rotvec.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    R[~safe] = np.eye(3)
    return R


def _matrix_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    sy = np.clip(-R[..., 2, 0], -1.0, 1.0)
    y = np.arcsin(sy)
    cy = np.cos(y)
    regular = np.abs(cy) > 1e-8
    x = np.zeros_like(y)
    z = np.zeros_like(y)
    x[regular] = np.arctan2(R[..., 2, 1][regular], R[..., 2, 2][regular])
    z[regular] = np.arctan2(R[..., 1, 0][regular], R[..., 0, 0][regular])
    x[~regular] = np.arctan2(-R[..., 1, 2][~regular], R[..., 1, 1][~regular])
    z[~regular] = 0.0
    return np.rad2deg(np.stack([x, y, z], axis=-1))


def _format_array(values: np.ndarray, integer: bool = False, per_line: int = 12) -> str:
    parts = [str(int(v)) if integer else f"{float(v):.9g}" for v in values]
    lines = []
    for i in range(0, len(parts), per_line):
        lines.append(",".join(parts[i : i + per_line]))
    return ",\n\t\t\t".join(lines)


def _replace_curve_values(block_text: str, values: np.ndarray) -> str:
    count = len(values)
    replacement = f"KeyValueFloat: *{count} {{\n\t\t\ta: {_format_array(values)}\n\t\t}}"
    return re.sub(
        r"KeyValueFloat:\s*\*\d+\s*\{\s*a:\s*.*?\n[ \t]*\}",
        replacement,
        block_text,
        count=1,
        flags=re.S,
    )


def npy_to_ascii_fbx(template_fbx: str, npy_path: str, output_fbx: str, verbose: bool = True) -> None:
    with open(template_fbx, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    payload, context = _assemble_motion(text, verbose=False)
    data = np.load(npy_path, allow_pickle=True).item()
    poses = np.asarray(data["poses"], dtype=np.float64).reshape(-1, 52, 3)
    trans = np.asarray(data["trans"], dtype=np.float64)
    if poses.shape[0] != payload["poses"].shape[0]:
        raise ValueError(
            f"模板帧数 {payload['poses'].shape[0]} 与 npy 帧数 {poses.shape[0]} 不一致"
        )

    eulers = _matrix_to_euler_xyz_deg(_axis_angle_to_matrix(poses))
    replacements: Dict[int, np.ndarray] = {}
    curves = context["curves"]
    curve_to_animnode = context["curve_to_animnode"]
    animnode_to_model = context["animnode_to_model"]
    models = context["models"]
    name_to_model = context["name_to_model"]
    joint_mapping = context["joint_mapping"]
    model_uid_to_joint = {
        name_to_model[fbx_name]["uid"]: ji
        for ji, joint in enumerate(SMPLX_JOINT_NAMES)
        for fbx_name in [joint_mapping.get(joint)]
        if fbx_name and fbx_name in name_to_model
    }
    pelvis_uid = name_to_model[joint_mapping["pelvis"]]["uid"] if "pelvis" in joint_mapping else None

    for cuid, (anuid, axis_name) in curve_to_animnode.items():
        if anuid not in animnode_to_model or cuid not in curves:
            continue
        muid, prop = animnode_to_model[anuid]
        axis_index = {"d|X": 0, "d|Y": 1, "d|Z": 2}[axis_name]
        if prop in ("Lcl Rotation", "LclRotation") and muid in model_uid_to_joint:
            replacements[cuid] = eulers[:, model_uid_to_joint[muid], axis_index]
        elif prop in ("Lcl Translation", "LclTranslation") and muid == pelvis_uid:
            replacements[cuid] = trans[:, axis_index] * 100.0

    chunks: List[str] = []
    cursor = 0
    replaced = 0
    for block in sorted(_iter_blocks(text, "AnimationCurve"), key=lambda item: item.start):
        chunks.append(text[cursor : block.start])
        block_text = text[block.start : block.end]
        if block.uid in replacements:
            block_text = _replace_curve_values(block_text, replacements[block.uid])
            replaced += 1
        chunks.append(block_text)
        cursor = block.end
    chunks.append(text[cursor:])
    out_text = "".join(chunks)

    out_dir = os.path.dirname(os.path.abspath(output_fbx))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_fbx, "w", encoding="utf-8", newline="\n") as f:
        f.write(out_text)
    if verbose:
        print(f"[Save] -> {output_fbx}")
        print(f"        已替换 {replaced} 条动画曲线")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    to_npy = sub.add_parser("to-npy")
    to_npy.add_argument("fbx")
    to_npy.add_argument("npy")
    to_npy.add_argument("--fps", type=float, default=None)
    to_npy.add_argument("--quiet", action="store_true")

    to_fbx = sub.add_parser("to-fbx")
    to_fbx.add_argument("template_fbx")
    to_fbx.add_argument("npy")
    to_fbx.add_argument("output_fbx")
    to_fbx.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    if args.cmd == "to-npy":
        ascii_fbx_to_npy(args.fbx, args.npy, fps_override=args.fps, verbose=not args.quiet)
    elif args.cmd == "to-fbx":
        npy_to_ascii_fbx(args.template_fbx, args.npy, args.output_fbx, verbose=not args.quiet)


if __name__ == "__main__":
    main()
