"""Web 任务工作进程：与 Flask 主进程隔离，避免 OOM 杀进程时网页服务一起退出。

向 stdout 输出 JSON 行（每行一个事件），主进程解析后更新前端状态。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def emit(obj: dict) -> None:
    # ensure_ascii=True：中文写成 \uXXXX，避免 Windows 管道 GBK/UTF-8 混用导致前端乱码
    sys.stdout.write(json.dumps(obj, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--character", default="silver_wolf_lv999")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 保证与 app 相同的设备策略（在 import app / torch 前由 app.apply_device_policy 处理）
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)

    try:
        import app as webapp
    except Exception:
        emit({"type": "error", "stage": 0, "error": traceback.format_exc()})
        return 2

    job = webapp.JobState(job_id=args.job_id, text=args.text, character=args.character)
    job.status = "running"

    def publish():
        snap = job.snapshot()
        snap["type"] = "snapshot"
        emit(snap)

    job.publish = publish  # type: ignore[method-assign]
    job.publish()

    try:
        webapp._execute_pipeline(job, args.text, args.frames, args.character)
        emit({"type": "done", **job.snapshot()})
        return 0
    except Exception:
        err = traceback.format_exc()
        stage_id = job.stage if job.stage >= 0 else 0
        job.status = "error"
        job.error = err
        job.message = f"阶段 {stage_id} 失败"
        for s in job.stages:
            if s["id"] == stage_id or s["status"] == "running":
                s["status"] = "error"
        job.publish()
        emit({"type": "error", "stage": stage_id, "error": err})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
