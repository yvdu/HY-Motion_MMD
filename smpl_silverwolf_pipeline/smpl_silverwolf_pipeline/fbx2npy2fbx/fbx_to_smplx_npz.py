"""
fbx_to_smplx_npz.py
====================
将 SMPLX 绑定的 FBX (Kaydara FBX Binary, 7.x) 转换为
Eco-HY-Motion 训练所用的 NPZ 格式：
    poses:            (T, 156) float32   axis-angle, 52 joints x 3
    trans:            (T, 3)  float32   根关节位移（单位：米）
    betas:            (16,)   float32   全 0 占位
    gender:           "neutral"
    mocap_framerate:  float            帧率

特点：
- 纯 Python 二进制 FBX 解析（无需 fbxsdkpy / bpy / Blender）
- 自动适配 SMPLH / SMPLX 常见骨骼命名（小写下划线 & 驼峰）
- 旋转：读取 LclRotation 欧拉（默认 XYZ 顺序，单位度）→ axis-angle
- 平移：读取 Pelvis 的 LclTranslation（厘米）→ 米（/100，与 smplh2woodfbx.py 保持一致）

用法：
    python fbx_to_smplx_npz.py /root/Human_56-1_03_SMPLX.fbx /root/Human_56-1_03_SMPLX.npz
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1) FBX 二进制解析器（公开格式：https://code.blender.org/2013/08/fbx-binary-file-format-specification/）
# ---------------------------------------------------------------------------

class FBXNode:
    __slots__ = ("name", "props", "children")

    def __init__(self, name: str = "", props: Optional[List[Any]] = None,
                 children: Optional[List["FBXNode"]] = None):
        self.name = name
        self.props = props if props is not None else []
        self.children = children if children is not None else []

    def find(self, name: str) -> Optional["FBXNode"]:
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name: str) -> List["FBXNode"]:
        return [c for c in self.children if c.name == name]


class FBXBinaryParser:
    """轻量 FBX 二进制解析器。"""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.buf = f.read()
        self.pos = 0
        # 校验 header
        if self.buf[:21] != b"Kaydara FBX Binary  \x00":
            raise ValueError("不是 Kaydara FBX Binary 文件")
        # bytes 21..22: 0x1A 0x00
        # bytes 23..26: version (little-endian uint32)
        self.version = struct.unpack_from("<I", self.buf, 23)[0]
        self.pos = 27
        # 7.5+ 使用 8 字节偏移；7.4 及以下使用 4 字节
        self.use_uint64 = self.version >= 7500

    # ----------- 读取原语 -----------
    def _read(self, fmt: str):
        size = struct.calcsize(fmt)
        v = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += size
        return v

    def _read_uint(self) -> int:
        if self.use_uint64:
            v = struct.unpack_from("<Q", self.buf, self.pos)[0]
            self.pos += 8
        else:
            v = struct.unpack_from("<I", self.buf, self.pos)[0]
            self.pos += 4
        return v

    def _read_array(self, elem_fmt: str, elem_size: int):
        length = struct.unpack_from("<I", self.buf, self.pos)[0]; self.pos += 4
        encoding = struct.unpack_from("<I", self.buf, self.pos)[0]; self.pos += 4
        comp_len = struct.unpack_from("<I", self.buf, self.pos)[0]; self.pos += 4
        raw = self.buf[self.pos:self.pos + comp_len]
        self.pos += comp_len
        if encoding == 1:
            raw = zlib.decompress(raw)
        # 解码为 numpy
        np_dtype = {
            'f': np.float32, 'd': np.float64,
            'i': np.int32, 'l': np.int64, 'b': np.int8,
        }[elem_fmt]
        arr = np.frombuffer(raw, dtype=np_dtype, count=length)
        return arr

    def _read_property(self):
        # 注意：为了 round-trip 时保留原始 type code（MotionBuilder 严格区分
        # I/L/Y/F/D），整数类型分别返回 np.int16/int32/int64，浮点分别返回
        # np.float32/float（python float 按 D=double 写出）。
        type_code = self.buf[self.pos:self.pos+1].decode('ascii'); self.pos += 1
        if type_code == 'Y':
            v, = self._read("<h"); return np.int16(v)
        if type_code == 'C':
            v, = self._read("<?"); return v
        if type_code == 'I':
            v, = self._read("<i"); return np.int32(v)
        if type_code == 'F':
            v, = self._read("<f"); return np.float32(v)
        if type_code == 'D':
            v, = self._read("<d"); return float(v)
        if type_code == 'L':
            v, = self._read("<q"); return np.int64(v)
        if type_code == 'f':
            return self._read_array('f', 4)
        if type_code == 'd':
            return self._read_array('d', 8)
        if type_code == 'l':
            return self._read_array('l', 8)
        if type_code == 'i':
            return self._read_array('i', 4)
        if type_code == 'b':
            return self._read_array('b', 1)
        if type_code == 'S':
            length, = self._read("<I")
            s = self.buf[self.pos:self.pos+length]; self.pos += length
            try:
                return s.decode('utf-8', errors='replace')
            except Exception:
                return s
        if type_code == 'R':
            length, = self._read("<I")
            data = self.buf[self.pos:self.pos+length]; self.pos += length
            return data
        raise ValueError(f"未知属性类型: {type_code!r} @ {self.pos}")

    def _read_node(self) -> Optional[FBXNode]:
        end_offset = self._read_uint()
        if end_offset == 0:
            return None
        num_props = self._read_uint()
        _prop_list_len = self._read_uint()
        name_len = self.buf[self.pos]; self.pos += 1
        name = self.buf[self.pos:self.pos+name_len].decode('ascii', errors='replace')
        self.pos += name_len
        props = [self._read_property() for _ in range(num_props)]
        children: List[FBXNode] = []
        # 子节点：直到达到 end_offset
        sentinel = 13 if self.use_uint64 else 13  # null record size: 3*uint + 1 byte
        if self.use_uint64:
            null_record_size = 25  # 3*8 + 1
        else:
            null_record_size = 13  # 3*4 + 1
        if self.pos < end_offset:
            while self.pos < end_offset - null_record_size:
                child = self._read_node()
                if child is None:
                    break
                children.append(child)
            # 跳过结束的 null record
            self.pos = end_offset
        return FBXNode(name, props, children)

    def parse(self) -> FBXNode:
        root = FBXNode("__root__", [], [])
        while self.pos < len(self.buf) - 1:
            child = self._read_node()
            if child is None:
                break
            root.children.append(child)
        return root


# ---------------------------------------------------------------------------
# 2) SMPLX 关节列表 / 命名映射
# ---------------------------------------------------------------------------

SMPLX_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]

# 备选命名（驼峰版本，来自 smplh2woodfbx.py）
CAMEL_NAMES = {
    "pelvis": "Pelvis", "left_hip": "L_Hip", "right_hip": "R_Hip", "spine1": "Spine1",
    "left_knee": "L_Knee", "right_knee": "R_Knee", "spine2": "Spine2",
    "left_ankle": "L_Ankle", "right_ankle": "R_Ankle", "spine3": "Spine3",
    "left_foot": "L_Foot", "right_foot": "R_Foot",
    "neck": "Neck", "left_collar": "L_Collar", "right_collar": "R_Collar", "head": "Head",
    "left_shoulder": "L_Shoulder", "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow", "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist", "right_wrist": "R_Wrist",
    "left_index1": "L_Index1", "left_index2": "L_Index2", "left_index3": "L_Index3",
    "left_middle1": "L_Middle1", "left_middle2": "L_Middle2", "left_middle3": "L_Middle3",
    "left_pinky1": "L_Pinky1", "left_pinky2": "L_Pinky2", "left_pinky3": "L_Pinky3",
    "left_ring1": "L_Ring1", "left_ring2": "L_Ring2", "left_ring3": "L_Ring3",
    "left_thumb1": "L_Thumb1", "left_thumb2": "L_Thumb2", "left_thumb3": "L_Thumb3",
    "right_index1": "R_Index1", "right_index2": "R_Index2", "right_index3": "R_Index3",
    "right_middle1": "R_Middle1", "right_middle2": "R_Middle2", "right_middle3": "R_Middle3",
    "right_pinky1": "R_Pinky1", "right_pinky2": "R_Pinky2", "right_pinky3": "R_Pinky3",
    "right_ring1": "R_Ring1", "right_ring2": "R_Ring2", "right_ring3": "R_Ring3",
    "right_thumb1": "R_Thumb1", "right_thumb2": "R_Thumb2", "right_thumb3": "R_Thumb3",
}


# ---------------------------------------------------------------------------
# 3) FBX 场景信息提取
# ---------------------------------------------------------------------------

def parse_fbx_object_name(raw: Any) -> str:
    """FBX 中 Model 的属性 1 通常是 "Model::name" 或 b"name\x00\x01Model"。"""
    if isinstance(raw, bytes):
        s = raw.decode('utf-8', errors='replace')
    else:
        s = str(raw)
    # FBX 二进制中常用分隔："name\x00\x01Model"
    if '\x00\x01' in s:
        s = s.split('\x00\x01')[0]
    # 文本形式 "Model::name"
    if "::" in s:
        s = s.split("::", 1)[1]
    return s.strip()


def extract_models(objects_node: FBXNode) -> Dict[int, Dict[str, Any]]:
    """提取所有 Model 节点，建立 uid -> 信息 的映射。"""
    models: Dict[int, Dict[str, Any]] = {}
    for m in objects_node.find_all("Model"):
        if len(m.props) < 3:
            continue
        uid = int(m.props[0])
        name = parse_fbx_object_name(m.props[1])
        mtype = parse_fbx_object_name(m.props[2]) if len(m.props) >= 3 else ""
        # 解析 LclTranslation / LclRotation / RotationOrder 等基础属性
        lcl_trans = np.zeros(3, dtype=np.float64)
        lcl_rot = np.zeros(3, dtype=np.float64)
        rotation_order = 0  # 0 = XYZ
        props70 = m.find("Properties70")
        if props70 is not None:
            for p in props70.find_all("P"):
                if len(p.props) < 1:
                    continue
                pname = p.props[0]
                if isinstance(pname, bytes):
                    pname = pname.decode('utf-8', errors='replace')
                if pname == "Lcl Translation" and len(p.props) >= 7:
                    lcl_trans = np.array([float(p.props[4]), float(p.props[5]), float(p.props[6])])
                elif pname == "Lcl Rotation" and len(p.props) >= 7:
                    lcl_rot = np.array([float(p.props[4]), float(p.props[5]), float(p.props[6])])
                elif pname == "RotationOrder" and len(p.props) >= 5:
                    rotation_order = int(p.props[4])
        models[uid] = {
            "uid": uid,
            "name": name,
            "type": mtype,
            "lcl_trans": lcl_trans,
            "lcl_rot": lcl_rot,
            "rotation_order": rotation_order,
        }
    return models


def extract_anim_curves(objects_node: FBXNode) -> Dict[int, Dict[str, Any]]:
    """提取 AnimationCurve（关键帧时间 / 值数组）。"""
    curves: Dict[int, Dict[str, Any]] = {}
    for c in objects_node.find_all("AnimationCurve"):
        if len(c.props) < 1:
            continue
        uid = int(c.props[0])
        kt = c.find("KeyTime")
        kv = c.find("KeyValueFloat")
        if kt is None or kv is None:
            continue
        times = kt.props[0] if len(kt.props) > 0 else None
        values = kv.props[0] if len(kv.props) > 0 else None
        if times is None or values is None:
            continue
        curves[uid] = {
            "uid": uid,
            "times": np.asarray(times, dtype=np.int64),   # ktime: 1/46186158000 秒
            "values": np.asarray(values, dtype=np.float64),
        }
    return curves


def extract_anim_curve_nodes(objects_node: FBXNode) -> Dict[int, Dict[str, Any]]:
    """提取 AnimationCurveNode（如 T/R/S）的默认值。"""
    nodes: Dict[int, Dict[str, Any]] = {}
    for n in objects_node.find_all("AnimationCurveNode"):
        if len(n.props) < 2:
            continue
        uid = int(n.props[0])
        name = parse_fbx_object_name(n.props[1])
        defaults = {"d|X": 0.0, "d|Y": 0.0, "d|Z": 0.0}
        props70 = n.find("Properties70")
        if props70 is not None:
            for p in props70.find_all("P"):
                if len(p.props) < 5:
                    continue
                pname = p.props[0]
                if isinstance(pname, bytes):
                    pname = pname.decode('utf-8', errors='replace')
                if pname in defaults:
                    try:
                        defaults[pname] = float(p.props[4])
                    except Exception:
                        pass
        nodes[uid] = {"uid": uid, "name": name, "defaults": defaults}
    return nodes


def extract_connections(root: FBXNode) -> List[Tuple[str, int, int, Optional[str]]]:
    """提取 Connections。返回 (type, src, dst, prop_name)。type: 'OO' 或 'OP'。"""
    out: List[Tuple[str, int, int, Optional[str]]] = []
    conn_node = root.find("Connections")
    if conn_node is None:
        return out
    for c in conn_node.find_all("C"):
        if len(c.props) < 3:
            continue
        ctype = c.props[0]
        if isinstance(ctype, bytes):
            ctype = ctype.decode('utf-8', errors='replace')
        ctype = str(ctype)
        # 仅处理 OO（对象->对象）/ OP（对象->属性）；忽略 PP 等
        if ctype not in ("OO", "OP"):
            continue
        try:
            src = int(c.props[1])
            dst = int(c.props[2])
        except (ValueError, TypeError):
            continue
        prop = None
        if ctype == "OP" and len(c.props) >= 4:
            prop = c.props[3]
            if isinstance(prop, bytes):
                prop = prop.decode('utf-8', errors='replace')
        out.append((ctype, src, dst, prop))
    return out


# ---------------------------------------------------------------------------
# 4) 旋转转换（欧拉 -> axis-angle）
# ---------------------------------------------------------------------------

def euler_xyz_deg_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
    """欧拉角 (X, Y, Z, 度) 按 XYZ 顺序（先 X 后 Y 最后 Z, 即 R = Rz Ry Rx, x' = Rz Ry Rx x）→ 旋转矩阵。
    与 transforms3d.euler.euler2mat(axes='sxyz') 等价。
    输入: (..., 3) 单位度。输出: (..., 3, 3)。
    """
    r = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    cx, cy, cz = np.cos(r[..., 0]), np.cos(r[..., 1]), np.cos(r[..., 2])
    sx, sy, sz = np.sin(r[..., 0]), np.sin(r[..., 1]), np.sin(r[..., 2])
    # Rx
    R_x = np.zeros(r.shape[:-1] + (3, 3))
    R_x[..., 0, 0] = 1
    R_x[..., 1, 1] = cx;  R_x[..., 1, 2] = -sx
    R_x[..., 2, 1] = sx;  R_x[..., 2, 2] = cx
    # Ry
    R_y = np.zeros_like(R_x)
    R_y[..., 0, 0] = cy;  R_y[..., 0, 2] = sy
    R_y[..., 1, 1] = 1
    R_y[..., 2, 0] = -sy; R_y[..., 2, 2] = cy
    # Rz
    R_z = np.zeros_like(R_x)
    R_z[..., 0, 0] = cz;  R_z[..., 0, 1] = -sz
    R_z[..., 1, 0] = sz;  R_z[..., 1, 1] = cz
    R_z[..., 2, 2] = 1
    # XYZ (sxyz): R = Rz @ Ry @ Rx
    return R_z @ R_y @ R_x


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> axis-angle (rotvec)。输入 (..., 3, 3)，输出 (..., 3)。"""
    # trace 计算 cos(theta)
    tr = np.trace(R, axis1=-2, axis2=-1)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    # 反对称部分
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    vec = np.stack([rx, ry, rz], axis=-1)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-8
    # 通用情况
    out = np.zeros_like(vec)
    safe = ~small
    if np.any(safe):
        out[safe] = vec[safe] * (theta[safe] / (2 * sin_theta[safe]))[..., None]
    # 小角度退化情形：theta ≈ 0 -> rotvec 0；theta ≈ pi -> 单独处理
    if np.any(small):
        idx = np.where(small)
        for ii in zip(*idx):
            Ri = R[ii]
            th = theta[ii]
            if th < 1e-6:
                out[ii] = 0.0
            else:
                # theta ≈ pi: axis 来自对称矩阵 (R + I)/2 的最大对角元
                M = (Ri + np.eye(3)) / 2.0
                diag = np.diag(M)
                k = int(np.argmax(diag))
                axis = M[:, k] / np.sqrt(max(diag[k], 1e-12))
                out[ii] = axis * th
    return out


# ---------------------------------------------------------------------------
# 5) 主流程
# ---------------------------------------------------------------------------

# FBX KTime: 1 second = 46186158000 units
FBX_KTIME_PER_SEC = 46186158000


def build_joint_mapping(model_names: List[str]) -> Dict[str, str]:
    """SMPLX joint name -> FBX node name."""
    name_set = set(model_names)
    mapping = {}
    for j in SMPLX_JOINT_NAMES:
        if j in name_set:
            mapping[j] = j
        elif CAMEL_NAMES[j] in name_set:
            mapping[j] = CAMEL_NAMES[j]
    return mapping


def sample_curve(curve: Optional[Dict[str, Any]], default: float, times: np.ndarray) -> np.ndarray:
    """根据采样时刻 times (单位 ktime) 在曲线上线性插值。"""
    if curve is None or len(curve["times"]) == 0:
        return np.full(times.shape, default, dtype=np.float64)
    ct = curve["times"].astype(np.float64)
    cv = curve["values"].astype(np.float64)
    return np.interp(times.astype(np.float64), ct, cv,
                     left=cv[0], right=cv[-1])


def convert(fbx_path: str, npz_path: str, fps_override: Optional[float] = None,
            verbose: bool = True) -> None:
    parser = FBXBinaryParser(fbx_path)
    if verbose:
        print(f"[FBX] version: {parser.version}, 64-bit offset: {parser.use_uint64}")
    root = parser.parse()

    # --- GlobalSettings: 帧率 ---
    fps = 30.0
    gs = root.find("GlobalSettings")
    if gs is not None:
        props70 = gs.find("Properties70")
        if props70 is not None:
            time_mode = None
            custom_fps = None
            for p in props70.find_all("P"):
                pname = p.props[0]
                if isinstance(pname, bytes):
                    pname = pname.decode('utf-8', errors='replace')
                if pname == "TimeMode" and len(p.props) >= 5:
                    time_mode = int(p.props[4])
                elif pname == "CustomFrameRate" and len(p.props) >= 5:
                    custom_fps = float(p.props[4])
            # TimeMode → fps（FBX 内置枚举的常见值）
            time_mode_map = {
                0: 30.0, 1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0,
                5: 48.0, 6: 30.0, 7: 30.0, 8: 29.97, 9: 29.97,
                10: 25.0, 11: 24.0, 12: 23.976, 13: 24.0, 14: -1.0,
                15: 96.0, 16: 72.0, 17: 59.94,
            }
            if time_mode is not None and time_mode in time_mode_map:
                tm_fps = time_mode_map[time_mode]
                if tm_fps > 0:
                    fps = tm_fps
            if custom_fps and custom_fps > 0 and (time_mode == 14 or fps == 30.0):
                fps = custom_fps
    if fps_override:
        fps = float(fps_override)
    if verbose:
        print(f"[FBX] fps: {fps}")

    # --- Objects ---
    objects_node = root.find("Objects")
    if objects_node is None:
        raise RuntimeError("FBX 中未找到 Objects 节点")
    models = extract_models(objects_node)
    curves = extract_anim_curves(objects_node)
    anim_nodes = extract_anim_curve_nodes(objects_node)
    connections = extract_connections(root)

    # 反向索引
    name_to_model: Dict[str, Dict[str, Any]] = {}
    for m in models.values():
        # 仅保留 LimbNode / Null / 普通骨骼（同名时优先 LimbNode）
        if m["name"] not in name_to_model or m["type"] in ("LimbNode", "Limb"):
            name_to_model[m["name"]] = m

    # connections: curve(src) -> anim_node(dst, prop="d|X/Y/Z")
    curve_to_animnode: Dict[int, Tuple[int, str]] = {}
    # anim_node -> model (prop = "Lcl Translation" / "Lcl Rotation" / "Lcl Scaling")
    animnode_to_model: Dict[int, Tuple[int, str]] = {}
    for ctype, src, dst, prop in connections:
        if ctype == "OP":
            if src in curves and dst in anim_nodes and prop in ("d|X", "d|Y", "d|Z"):
                curve_to_animnode[src] = (dst, prop)
            elif src in anim_nodes and dst in models and prop:
                animnode_to_model[src] = (dst, prop)

    # 组装：model_uid -> { "T": {X,Y,Z curves}, "R": {X,Y,Z curves} }
    model_curves: Dict[int, Dict[str, Dict[str, Optional[Dict[str, Any]]]]] = {}
    # 初始化默认值
    for muid in models.keys():
        model_curves[muid] = {
            "T": {"X": None, "Y": None, "Z": None,
                  "defaults": models[muid]["lcl_trans"].tolist()},
            "R": {"X": None, "Y": None, "Z": None,
                  "defaults": models[muid]["lcl_rot"].tolist()},
        }
    # 反向遍历曲线
    for cuid, (anuid, axis) in curve_to_animnode.items():
        if anuid not in animnode_to_model:
            continue
        muid, prop = animnode_to_model[anuid]
        if muid not in model_curves:
            continue
        ax = axis[-1]  # 'X'/'Y'/'Z'
        if prop in ("Lcl Translation", "LclTranslation"):
            model_curves[muid]["T"][ax] = curves[cuid]
        elif prop in ("Lcl Rotation", "LclRotation"):
            model_curves[muid]["R"][ax] = curves[cuid]
        # 也用 AnimNode 的 d|? 默认值覆盖 model 的默认值
        an_default = anim_nodes[anuid]["defaults"].get(axis, None)
        if an_default is not None:
            ai = {"X": 0, "Y": 1, "Z": 2}[ax]
            if prop in ("Lcl Translation", "LclTranslation"):
                model_curves[muid]["T"]["defaults"][ai] = an_default
            elif prop in ("Lcl Rotation", "LclRotation"):
                model_curves[muid]["R"]["defaults"][ai] = an_default

    # --- 关节映射 ---
    joint_mapping = build_joint_mapping(list(name_to_model.keys()))
    if verbose:
        print(f"[Map] matched {len(joint_mapping)} / 52 SMPLX joints")
        missing = [j for j in SMPLX_JOINT_NAMES if j not in joint_mapping]
        if missing:
            print(f"[Map] missing: {missing[:8]}{'...' if len(missing) > 8 else ''}")

    # --- 计算时间轴 ---
    # 收集所有曲线的时间范围
    all_times: List[np.ndarray] = []
    for j in SMPLX_JOINT_NAMES:
        fbx_name = joint_mapping.get(j)
        if not fbx_name:
            continue
        muid = name_to_model[fbx_name]["uid"]
        for ax in ("X", "Y", "Z"):
            for tr in ("T", "R"):
                cv = model_curves[muid][tr][ax]
                if cv is not None and len(cv["times"]) > 0:
                    all_times.append(cv["times"])
    if not all_times:
        raise RuntimeError("未找到任何动画曲线")
    t0 = min(int(t.min()) for t in all_times)
    t1 = max(int(t.max()) for t in all_times)
    duration_sec = (t1 - t0) / FBX_KTIME_PER_SEC
    num_frames = max(1, int(round(duration_sec * fps)) + 1)
    sample_times = np.linspace(t0, t1, num_frames).astype(np.int64)
    if verbose:
        print(f"[Anim] frames: {num_frames}, duration: {duration_sec:.3f}s")

    # --- 对每个关节采样欧拉旋转，并转 axis-angle ---
    poses = np.zeros((num_frames, 52, 3), dtype=np.float64)
    trans = np.zeros((num_frames, 3), dtype=np.float64)

    for ji, jname in enumerate(SMPLX_JOINT_NAMES):
        fbx_name = joint_mapping.get(jname)
        if not fbx_name:
            continue
        muid = name_to_model[fbx_name]["uid"]
        rblock = model_curves[muid]["R"]
        defaults_r = rblock["defaults"]
        eulers = np.zeros((num_frames, 3), dtype=np.float64)
        for ai, ax in enumerate(("X", "Y", "Z")):
            eulers[:, ai] = sample_curve(rblock[ax], defaults_r[ai], sample_times)
        # 欧拉(度) -> 矩阵 -> axis-angle
        Rm = euler_xyz_deg_to_matrix(eulers)
        poses[:, ji, :] = matrix_to_axis_angle(Rm)

    # --- 根关节平移（Pelvis） ---
    if "pelvis" in joint_mapping:
        muid = name_to_model[joint_mapping["pelvis"]]["uid"]
        tblock = model_curves[muid]["T"]
        defaults_t = tblock["defaults"]
        for ai, ax in enumerate(("X", "Y", "Z")):
            trans[:, ai] = sample_curve(tblock[ax], defaults_t[ai], sample_times)
        # 厘米 -> 米（与 smplh2woodfbx.py 的 scale=100 反向一致）
        trans /= 100.0

    # --- 保存为 NPZ ---
    poses_flat = poses.reshape(num_frames, -1).astype(np.float32)
    out_dir = os.path.dirname(os.path.abspath(npz_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        npz_path,
        poses=poses_flat,                                    # (T, 156) float32
        trans=trans.astype(np.float32),                      # (T, 3)   float32 (米)
        betas=np.zeros(16, dtype=np.float32),                # (16,)    占位
        gender="neutral",
        mocap_framerate=np.float32(fps),
    )
    if verbose:
        print(f"[Save] -> {npz_path}")
        print(f"        poses {poses_flat.shape}, trans {trans.shape}, fps {fps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fbx", nargs="?", default="/root/Human_56-1_03_SMPLX.fbx",
                    help="输入 FBX 文件路径")
    ap.add_argument("npz", nargs="?", default="/root/Human_56-1_03_SMPLX.npz",
                    help="输出 NPZ 文件路径")
    ap.add_argument("--fps", type=float, default=None, help="覆盖帧率")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    convert(args.fbx, args.npz, fps_override=args.fps, verbose=not args.quiet)


if __name__ == "__main__":
    main()
