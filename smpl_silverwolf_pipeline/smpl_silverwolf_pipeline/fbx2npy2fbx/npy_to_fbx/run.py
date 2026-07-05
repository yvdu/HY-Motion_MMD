"""批量把项目约定的 SMPLX motion .npy/.npz 写回 ASCII FBX。"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ascii_fbx_smplx_io import npy_to_ascii_fbx


def 读取配置(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def 列出_motion(input_path: str, recursive: bool) -> List[str]:
    input_path = os.path.abspath(input_path)
    suffixes = (".npy", ".npz")
    if os.path.isfile(input_path):
        return [input_path] if input_path.lower().endswith(suffixes) else []

    out: List[str] = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(input_path):
            for filename in filenames:
                if filename.lower().endswith(suffixes):
                    out.append(os.path.join(dirpath, filename))
    else:
        for filename in os.listdir(input_path):
            path = os.path.join(input_path, filename)
            if os.path.isfile(path) and filename.lower().endswith(suffixes):
                out.append(path)
    out.sort()
    return out


def 生成输出路径(motion_path: str, input_path: str, output_dir: str, preserve_subdirs: bool) -> str:
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    if os.path.isfile(input_path) or not preserve_subdirs:
        rel_path = os.path.basename(motion_path)
    else:
        rel_path = os.path.relpath(motion_path, input_path)
    rel_path = os.path.splitext(rel_path)[0] + ".fbx"
    return os.path.join(output_dir, rel_path)


def 查找模板路径(
    motion_path: str,
    input_path: str,
    template_fbx: str | None,
    template_dir: str | None,
) -> str:
    if template_dir:
        input_path = os.path.abspath(input_path)
        if os.path.isfile(input_path):
            rel_path = os.path.basename(motion_path)
        else:
            rel_path = os.path.relpath(motion_path, input_path)
        candidate = os.path.join(os.path.abspath(template_dir), os.path.splitext(rel_path)[0] + ".fbx")
        if os.path.exists(candidate):
            return candidate

    if template_fbx and os.path.exists(template_fbx):
        return os.path.abspath(template_fbx)

    raise FileNotFoundError(f"没有找到可用模板 FBX: motion={motion_path}")


def npz_转临时_npy(npz_path: str) -> str:
    with np.load(npz_path, allow_pickle=True) as data:
        payload = {
            "poses": np.asarray(data["poses"], dtype=np.float32),
            "trans": np.asarray(data["trans"], dtype=np.float32),
            "betas": np.asarray(
                data["betas"] if "betas" in data.files else np.zeros(16, dtype=np.float32),
                dtype=np.float32,
            ),
            "gender": str(data["gender"] if "gender" in data.files else "neutral"),
            "mocap_framerate": np.float32(
                data["mocap_framerate"] if "mocap_framerate" in data.files else 30.0
            ),
        }
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    tmp.close()
    np.save(tmp.name, np.array(payload, dtype=object), allow_pickle=True)
    return tmp.name


def 转换单个任务(task: Tuple[str, str, str, bool]) -> Tuple[str, bool, str]:
    motion_path, template_path, output_path, overwrite = task
    tmp_npy: str | None = None
    try:
        if (not overwrite) and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, True, "跳过：输出已存在"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        source_path = motion_path
        if motion_path.lower().endswith(".npz"):
            tmp_npy = npz_转临时_npy(motion_path)
            source_path = tmp_npy

        npy_to_ascii_fbx(template_path, source_path, output_path, verbose=False)
        return output_path, True, "完成"
    except Exception as exc:  # noqa: BLE001
        return output_path, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        if tmp_npy:
            try:
                os.remove(tmp_npy)
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="批量把 SMPLX motion .npy/.npz 写回 ASCII FBX")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="配置文件路径",
    )
    args = parser.parse_args()

    config = 读取配置(args.config)
    input_path = os.path.abspath(str(config["input_path"]))
    output_dir = os.path.abspath(str(config["output_dir"]))
    recursive = bool(config.get("recursive", True))
    preserve_subdirs = bool(config.get("preserve_subdirs", True))
    overwrite = bool(config.get("overwrite", False))
    template_fbx = config.get("template_fbx")
    template_dir = config.get("template_dir")
    workers = max(1, int(config.get("workers", 1)))

    motion_files = 列出_motion(input_path, recursive)
    tasks = []
    for motion_path in motion_files:
        output_path = 生成输出路径(motion_path, input_path, output_dir, preserve_subdirs)
        template_path = 查找模板路径(motion_path, input_path, template_fbx, template_dir)
        tasks.append((motion_path, template_path, output_path, overwrite))

    print(f"[扫描] 输入: {input_path}")
    print(f"[扫描] 找到 NPY/NPZ: {len(tasks)} 个")
    print(f"[运行] 输出目录: {output_dir}")
    print(f"[运行] workers={workers}, overwrite={overwrite}")

    ok_count = 0
    skip_count = 0
    fail_count = 0
    os.makedirs(output_dir, exist_ok=True)
    fail_log = os.path.join(output_dir, "_failed_npy_to_fbx.log")

    if workers == 1:
        iterator = (转换单个任务(task) for task in tasks)
        for index, (output_path, ok, message) in enumerate(iterator, 1):
            if ok and message.startswith("跳过"):
                skip_count += 1
            elif ok:
                ok_count += 1
            else:
                fail_count += 1
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(f"失败 {output_path}\n{message}\n\n")
            print(f"[{index}/{len(tasks)}] {message}: {output_path}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(转换单个任务, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                output_path, ok, message = future.result()
                if ok and message.startswith("跳过"):
                    skip_count += 1
                elif ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    with open(fail_log, "a", encoding="utf-8") as f:
                        f.write(f"失败 {output_path}\n{message}\n\n")
                print(f"[{index}/{len(tasks)}] {message}: {output_path}")

    print(f"[完成] 成功={ok_count}, 跳过={skip_count}, 失败={fail_count}")
    if fail_count:
        print(f"[完成] 失败日志: {fail_log}")


if __name__ == "__main__":
    main()
