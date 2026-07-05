"""批量把 FBX 转成项目约定的 SMPLX motion .npy。"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, List, Tuple

import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ascii_fbx_smplx_io import ascii_fbx_to_npy
from batch_fbx_to_smplx_npy import fbx_to_npy as binary_fbx_to_npy


def 读取配置(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def 自动进程数() -> int:
    """根据机器核心数给出较保守的默认进程数，避免把磁盘读写打满。"""
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    return max(1, min(cpu_count - 1, 8))


def 解析进程数(value: Any) -> int:
    """workers 为 0、null 或 auto 时自动选择进程数。"""
    if value is None:
        return 自动进程数()
    if isinstance(value, str) and value.strip().lower() == "auto":
        return 自动进程数()
    workers = int(value)
    return 自动进程数() if workers <= 0 else max(1, workers)


def 是否二进制_fbx(fbx_path: str) -> bool:
    with open(fbx_path, "rb") as f:
        return f.read(21) == b"Kaydara FBX Binary  \x00"


def 列出_fbx(input_path: str, recursive: bool) -> List[str]:
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return [input_path] if input_path.lower().endswith(".fbx") else []

    out: List[str] = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(input_path):
            for filename in filenames:
                if filename.lower().endswith(".fbx"):
                    out.append(os.path.join(dirpath, filename))
    else:
        for filename in os.listdir(input_path):
            path = os.path.join(input_path, filename)
            if os.path.isfile(path) and filename.lower().endswith(".fbx"):
                out.append(path)
    out.sort()
    return out


def 生成输出路径(fbx_path: str, input_path: str, output_dir: str, preserve_subdirs: bool) -> str:
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    if os.path.isfile(input_path) or not preserve_subdirs:
        rel_path = os.path.basename(fbx_path)
    else:
        rel_path = os.path.relpath(fbx_path, input_path)
    rel_path = os.path.splitext(rel_path)[0] + ".npy"
    return os.path.join(output_dir, rel_path)


def 任务排序键(task: Tuple[str, str, float | None, bool]) -> int:
    """大文件优先提交，能减少并行处理尾部只剩大文件的等待时间。"""
    try:
        return os.path.getsize(task[0])
    except OSError:
        return 0


def 转换单个任务(task: Tuple[str, str, float | None, bool]) -> Tuple[str, bool, str, float]:
    start_time = time.perf_counter()
    fbx_path, npy_path, fps_override, overwrite = task
    try:
        if (not overwrite) and os.path.exists(npy_path) and os.path.getsize(npy_path) > 0:
            return npy_path, True, "跳过：输出已存在", time.perf_counter() - start_time

        os.makedirs(os.path.dirname(os.path.abspath(npy_path)), exist_ok=True)
        if 是否二进制_fbx(fbx_path):
            _path, ok, message = binary_fbx_to_npy(
                fbx_path,
                npy_path,
                fps_override=fps_override,
                overwrite=True,
            )
            return npy_path, ok, message, time.perf_counter() - start_time

        ascii_fbx_to_npy(fbx_path, npy_path, fps_override=fps_override, verbose=False)
        return npy_path, True, "完成", time.perf_counter() - start_time
    except Exception as exc:  # noqa: BLE001
        return (
            npy_path,
            False,
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            time.perf_counter() - start_time,
        )


def 记录结果(
    result: Tuple[str, bool, str, float],
    fail_log: str,
) -> Tuple[int, int, int]:
    output_path, ok, message, _elapsed = result
    if ok and message.startswith("跳过"):
        return 0, 1, 0
    if ok:
        return 1, 0, 0

    with open(fail_log, "a", encoding="utf-8") as f:
        f.write(f"失败 {output_path}\n{message}\n\n")
    return 0, 0, 1


def 并行运行(
    tasks: List[Tuple[str, str, float | None, bool]],
    workers: int,
    chunksize: int,
) -> Iterable[Tuple[str, bool, str, float]]:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(转换单个任务, tasks, chunksize=max(1, chunksize))


def main() -> None:
    parser = argparse.ArgumentParser(description="批量把 FBX 转成 SMPLX motion .npy")
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
    fps_override = config.get("fps_override", None)
    workers = 解析进程数(config.get("workers", "auto"))
    chunksize = max(1, int(config.get("chunksize", 4)))
    progress_interval = max(1, int(config.get("progress_interval", 50)))

    fbx_files = 列出_fbx(input_path, recursive)
    tasks = [
        (
            fbx_path,
            生成输出路径(fbx_path, input_path, output_dir, preserve_subdirs),
            fps_override,
            overwrite,
        )
        for fbx_path in fbx_files
    ]
    tasks.sort(key=任务排序键, reverse=True)

    print(f"[扫描] 输入: {input_path}")
    print(f"[扫描] 找到 FBX: {len(tasks)} 个")
    print(f"[运行] 输出目录: {output_dir}")
    print(f"[运行] workers={workers}, chunksize={chunksize}, overwrite={overwrite}")

    ok_count = 0
    skip_count = 0
    fail_count = 0
    os.makedirs(output_dir, exist_ok=True)
    fail_log = os.path.join(output_dir, "_failed_fbx_to_npy.log")
    if os.path.exists(fail_log):
        os.remove(fail_log)

    start_time = time.perf_counter()
    if workers == 1:
        iterator = (转换单个任务(task) for task in tasks)
        for index, result in enumerate(iterator, 1):
            ok_delta, skip_delta, fail_delta = 记录结果(result, fail_log)
            ok_count += ok_delta
            skip_count += skip_delta
            fail_count += fail_delta
            if index % progress_interval == 0 or index == len(tasks):
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                speed = index / elapsed
                remaining = (len(tasks) - index) / speed if speed > 0 else 0.0
                print(
                    f"[进度] {index}/{len(tasks)} 成功={ok_count} 跳过={skip_count} "
                    f"失败={fail_count} 速度={speed:.2f}个/秒 剩余约={remaining/60:.1f}分钟"
                )
    else:
        for index, result in enumerate(并行运行(tasks, workers, chunksize), 1):
            ok_delta, skip_delta, fail_delta = 记录结果(result, fail_log)
            ok_count += ok_delta
            skip_count += skip_delta
            fail_count += fail_delta
            if index % progress_interval == 0 or index == len(tasks):
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                speed = index / elapsed
                remaining = (len(tasks) - index) / speed if speed > 0 else 0.0
                print(
                    f"[进度] {index}/{len(tasks)} 成功={ok_count} 跳过={skip_count} "
                    f"失败={fail_count} 速度={speed:.2f}个/秒 剩余约={remaining/60:.1f}分钟"
                )

    total_elapsed = time.perf_counter() - start_time
    print(f"[完成] 成功={ok_count}, 跳过={skip_count}, 失败={fail_count}")
    print(f"[完成] 总耗时={total_elapsed/60:.2f}分钟，平均速度={len(tasks)/max(total_elapsed, 1e-6):.2f}个/秒")
    if fail_count:
        print(f"[完成] 失败日志: {fail_log}")


if __name__ == "__main__":
    main()
