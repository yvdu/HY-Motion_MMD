"""
batch_fbx_to_smplx_npy.py
=========================
批量把 SuSu 数据集里的 FBX 转成 .npy（dict 形式，键与 fbx_to_smplx_npz.py
输出的 NPZ 一致：poses / trans / betas / gender / mocap_framerate）。

输入根目录： /apdcephfs_cq11/share_4502729/dataset/SuSu_to_SMPL
输出根目录： /apdcephfs_cq11/share_4502729/dataset/SuSu_to_SMPL_npy
子目录结构保持一致；输出文件名与输入同名，仅后缀 .fbx -> .npy。

实现思路：
- 复用 fbx_to_smplx_npz.py 的解析与提取逻辑；
- 在 convert(...) 里 NPZ 已经包含 (poses, trans, betas, gender, mocap_framerate)，
  这里把内部计算复制到一个 helper 中，直接得到这些数组并 np.save 成 .npy；
- 最简实现：调用 convert() 写到一个 NamedTemporaryFile.npz，再读回来转存成 .npy。
  避免重复实现解析逻辑、与 fbx_to_smplx_npz.py 行为完全一致。

用法：
    python batch_fbx_to_smplx_npy.py
    python batch_fbx_to_smplx_npy.py \
        --src /apdcephfs_cq11/share_4502729/dataset/SuSu_to_SMPL \
        --dst /apdcephfs_cq11/share_4502729/dataset/SuSu_to_SMPL_npy \
        --workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbx_to_smplx_npz import convert  # noqa: E402


DEFAULT_SRC = "/apdcephfs_cq11/share_4502729/dataset/SuSu_to_SMPL"
DEFAULT_DST = "/apdcephfs_cq11/share_4502729/dataset/SuSu_to_SMPL_npy"


def list_fbx_files(src_root: str) -> List[str]:
    """递归列出 src_root 下所有 .fbx 文件（不区分大小写）。"""
    out: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(src_root):
        for fn in filenames:
            if fn.lower().endswith(".fbx"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def fbx_to_npy(fbx_path: str, npy_path: str, fps_override: float | None = None,
               overwrite: bool = False) -> Tuple[str, bool, str]:
    """单文件转换：FBX -> .npy（dict 格式，与 NPZ 同字段）。

    返回 (npy_path, success, message)
    """
    try:
        if (not overwrite) and os.path.exists(npy_path) and os.path.getsize(npy_path) > 0:
            return (npy_path, True, "skip-existing")

        os.makedirs(os.path.dirname(os.path.abspath(npy_path)), exist_ok=True)

        # 1) 调用现成的 convert() 写到临时 npz
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
            tmp_npz = tf.name
        try:
            convert(fbx_path, tmp_npz, fps_override=fps_override, verbose=False)

            # 2) 读回来组成 dict 后用 np.save 写成 .npy
            with np.load(tmp_npz, allow_pickle=True) as data:
                payload = {
                    "poses": np.asarray(data["poses"]),                  # (T, 156) float32
                    "trans": np.asarray(data["trans"]),                  # (T, 3)   float32  米
                    "betas": np.asarray(data["betas"]),                  # (16,)    float32
                    "gender": str(data["gender"]),                       # 标量
                    "mocap_framerate": np.float32(data["mocap_framerate"]),
                }
            # 用 0-d object array 包 dict，加载侧 np.load(..., allow_pickle=True).item() 即可
            np.save(npy_path, np.array(payload, dtype=object), allow_pickle=True)
        finally:
            try:
                os.remove(tmp_npz)
            except OSError:
                pass

        return (npy_path, True, "ok")
    except Exception as e:  # noqa: BLE001
        return (npy_path, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def _worker(args: Tuple[str, str, float | None, bool]) -> Tuple[str, bool, str]:
    fbx, npy, fps, overwrite = args
    return fbx_to_npy(fbx, npy, fps_override=fps, overwrite=overwrite)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="FBX 根目录")
    ap.add_argument("--dst", default=DEFAULT_DST, help="输出 .npy 根目录")
    ap.add_argument("--fps", type=float, default=None, help="覆盖帧率（默认按 FBX）")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                    help="并行进程数，默认 CPU 核数")
    ap.add_argument("--overwrite", action="store_true", help="已存在的 .npy 也重新生成")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 个（调试用），0 表示全部")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要处理的文件数量，不实际转换")
    args = ap.parse_args()

    src_root = os.path.abspath(args.src)
    dst_root = os.path.abspath(args.dst)
    if not os.path.isdir(src_root):
        raise SystemExit(f"src 不存在或不是目录: {src_root}")

    print(f"[Scan] {src_root}")
    fbx_list = list_fbx_files(src_root)
    print(f"[Scan] 共 {len(fbx_list)} 个 FBX")

    if args.limit > 0:
        fbx_list = fbx_list[:args.limit]
        print(f"[Scan] 调试模式，仅处理前 {len(fbx_list)} 个")

    # 构造任务列表：保持子目录结构
    tasks: List[Tuple[str, str, float | None, bool]] = []
    for fbx in fbx_list:
        rel = os.path.relpath(fbx, src_root)
        rel_npy = os.path.splitext(rel)[0] + ".npy"
        npy_path = os.path.join(dst_root, rel_npy)
        tasks.append((fbx, npy_path, args.fps, args.overwrite))

    if args.dry_run:
        for fbx, npy, _, _ in tasks[:10]:
            print(f"  {fbx}\n   -> {npy}")
        if len(tasks) > 10:
            print(f"  ... 等共 {len(tasks)} 项")
        return

    print(f"[Run]  workers={args.workers}, overwrite={args.overwrite}")
    n_ok = 0
    n_skip = 0
    n_fail = 0
    fail_log = os.path.join(dst_root, "_failed.log")
    os.makedirs(dst_root, exist_ok=True)
    flog = open(fail_log, "a", encoding="utf-8")

    try:
        if args.workers <= 1:
            iterator = (_worker(t) for t in tasks)
            total = len(tasks)
            for i, (npy, ok, msg) in enumerate(iterator, 1):
                if ok and msg == "skip-existing":
                    n_skip += 1
                elif ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    flog.write(f"FAIL {npy}\n{msg}\n\n")
                    flog.flush()
                if i % 50 == 0 or i == total:
                    print(f"  [{i}/{total}] ok={n_ok} skip={n_skip} fail={n_fail}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_worker, t): t for t in tasks}
                total = len(futures)
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    npy, ok, msg = fut.result()
                    if ok and msg == "skip-existing":
                        n_skip += 1
                    elif ok:
                        n_ok += 1
                    else:
                        n_fail += 1
                        flog.write(f"FAIL {npy}\n{msg}\n\n")
                        flog.flush()
                    if done % 50 == 0 or done == total:
                        print(f"  [{done}/{total}] ok={n_ok} skip={n_skip} fail={n_fail}")
    finally:
        flog.close()

    print(f"[Done] total={len(tasks)} ok={n_ok} skip={n_skip} fail={n_fail}")
    if n_fail:
        print(f"[Done] 失败列表写入: {fail_log}")


if __name__ == "__main__":
    main()
