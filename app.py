"""文本生成 MMD 动作 DEMO 本地网页应用。

启动后自动打开浏览器，输入文本即可跑完整 pipeline，并展示阶段进度、
推理进度条、错误信息与最终渲染视频。

显存策略：启动时检测 GPU 总显存，<30GB 则 CPU 推理并在前端提示；
≥30GB 则使用 GPU。

用法:
  conda activate hymotion-mmd
  $env:USE_HF_MODELS="0"
  python app.py
"""
from __future__ import annotations

import os
import os.path as osp
import subprocess
import sys

# --------------------------------------------------------------------------- #
#  路径 / 环境 —— 先探测显存，再决定是否在 import torch 前隐藏 GPU
# --------------------------------------------------------------------------- #
REPO_ROOT = osp.dirname(osp.abspath(__file__))
PKG_ROOT = osp.join(REPO_ROOT, "smpl_silverwolf_pipeline", "smpl_silverwolf_pipeline")
CONFIG_PATH = osp.join(PKG_ROOT, "pipeline_config.yaml")
RENDER_SCRIPT = osp.join(REPO_ROOT, "scripts", "render_pmx_vmd.py")

# 官方 HY-Motion 约需 26GB；阈值用 30GB，前端提示文案用 26G
VRAM_GPU_THRESHOLD_GB = 30.0
VRAM_NOTICE_GB = 26

os.environ.setdefault("USE_HF_MODELS", "0")

CHARACTERS = [
    {"id": "silver_wolf_lv999", "label": "银狼 LV.999"},
]

DEVICE_INFO = {
    "use_cpu": True,
    "device": "cpu",
    "vram_gb": None,
    "gpu_name": None,
    "notice": "",
}


def probe_vram_gb():
    """返回 (总显存 GiB, GPU 名称)。优先 nvidia-smi，避免过早 import torch。"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        )
        best_gb, best_name = None, None
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue
            mem_mib = float(parts[0])
            name = parts[1] if len(parts) > 1 else None
            gb = mem_mib / 1024.0
            if best_gb is None or gb > best_gb:
                best_gb, best_name = gb, name
        return best_gb, best_name
    except Exception:
        pass

    # 回退：子进程 import torch，避免污染当前进程后再切 CPU
    try:
        code = (
            "import torch\n"
            "if torch.cuda.is_available():\n"
            "  i=0\n"
            "  print(torch.cuda.get_device_properties(i).total_memory/1024**3)\n"
            "  print(torch.cuda.get_device_name(i))\n"
            "else:\n"
            "  print('none')\n"
        )
        out = subprocess.check_output(
            [sys.executable, "-c", code],
            text=True,
            timeout=60,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        if out and out[0] != "none":
            return float(out[0]), (out[1] if len(out) > 1 else None)
    except Exception:
        pass
    return None, None


def probe_ram_gb():
    """系统物理内存 GiB。"""
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return int(out) / (1024**3)
    except Exception:
        pass
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys / (1024**3)
    except Exception:
        pass
    return None


def apply_device_policy():
    """按显存选择 CPU/GPU，并在需要时于 import torch 前隐藏 GPU（Windows 用 -1）。"""
    global DEVICE_INFO
    vram_gb, gpu_name = probe_vram_gb()
    ram_gb = probe_ram_gb()
    use_cpu = vram_gb is None or vram_gb < VRAM_GPU_THRESHOLD_GB

    if use_cpu:
        os.environ["HY_MOTION_DEVICE"] = "cpu"
        # Windows 上空字符串无效
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["HIP_VISIBLE_DEVICES"] = "-1"
        if vram_gb is None:
            notice = "未检测到可用 GPU，采用 CPU 推理（必须加载 Qwen，较慢）"
        else:
            notice = f"显存小于{VRAM_NOTICE_GB}G，采用CPU推理（必须加载 Qwen，磁盘卸载，较慢）"
        if ram_gb is not None and ram_gb < 24:
            notice += f"；系统内存约 {ram_gb:.0f}GB，请关闭其它程序以免 OOM"
    else:
        os.environ["HY_MOTION_DEVICE"] = "cuda"
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        notice = f"检测到显存约 {vram_gb:.1f} GB，使用 GPU 推理"

    DEVICE_INFO = {
        "use_cpu": use_cpu,
        "device": "cpu" if use_cpu else "cuda",
        "vram_gb": None if vram_gb is None else round(vram_gb, 2),
        "ram_gb": None if ram_gb is None else round(ram_gb, 1),
        "gpu_name": gpu_name,
        "notice": notice,
        "threshold_gb": VRAM_GPU_THRESHOLD_GB,
    }
    return DEVICE_INFO


# 必须在导入会间接加载 torch 的模块之前执行
apply_device_policy()

import importlib.util
import json
import queue
import random
import shutil
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PKG_ROOT)

# 延迟导入 pipeline 模块
def _load_run_pipeline():
    path = osp.join(PKG_ROOT, "run_pipeline.py")
    spec = importlib.util.spec_from_file_location("smpl_run_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


rp = _load_run_pipeline()

STAGE_DEFS = [
    {"id": 0, "name": "HY-Motion 文本生成动作", "key": "infer"},
    {"id": 1, "name": "SMPL FBX 转换", "key": "smpl_fbx"},
    {"id": 2, "name": "银狼动作重定向", "key": "retarget"},
    {"id": 3, "name": "ASCII → 二进制 FBX", "key": "binary"},
    {"id": 4, "name": "导出 VMD", "key": "vmd"},
    {"id": 5, "name": "渲染视频 (PMX+VMD)", "key": "render"},
]


# --------------------------------------------------------------------------- #
#  任务状态
# --------------------------------------------------------------------------- #
@dataclass
class JobState:
    job_id: str
    text: str
    character: str = "silver_wolf_lv999"
    status: str = "queued"  # queued|running|done|error
    stage: int = -1
    stage_name: str = ""
    stages: List[Dict[str, Any]] = field(default_factory=list)
    infer_step: int = 0
    infer_total: int = 0
    message: str = ""
    error: str = ""
    video_url: str = ""
    vmd_path: str = ""
    pmx_path: str = ""
    events: queue.Queue = field(default_factory=queue.Queue)

    def __post_init__(self):
        if not self.stages:
            self.stages = [
                {"id": s["id"], "name": s["name"], "status": "pending"} for s in STAGE_DEFS
            ]

    def snapshot(self) -> dict:
        return {
            "job_id": self.job_id,
            "text": self.text,
            "character": self.character,
            "status": self.status,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "stages": self.stages,
            "infer_step": self.infer_step,
            "infer_total": self.infer_total,
            "message": self.message,
            "error": self.error,
            "video_url": self.video_url,
            "vmd_path": self.vmd_path,
            "pmx_path": self.pmx_path,
            "device": DEVICE_INFO.get("device"),
            "use_cpu": DEVICE_INFO.get("use_cpu"),
        }

    def publish(self):
        self.events.put(self.snapshot())


JOBS: Dict[str, JobState] = {}
JOBS_LOCK = threading.Lock()
_runtime = None
_runtime_lock = threading.Lock()
_run_lock = threading.Lock()  # 同时只跑一个任务（模型占内存大）


def get_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        from hymotion.utils.t2m_runtime import T2MRuntime

        cfg = rp.load_config(CONFIG_PATH)
        hm = cfg.get("hymotion", {}) or {}
        model_path = rp.resolve(hm.get("model_path", "ckpts/tencent/HY-Motion-1.0"))
        config_yml = osp.join(model_path, "config.yml")
        ckpt = osp.join(model_path, "latest.ckpt")
        force_cpu = bool(DEVICE_INFO.get("use_cpu")) or os.environ.get("HY_MOTION_DEVICE", "").lower() == "cpu"
        print(
            f">>> Loading T2MRuntime (force_cpu={force_cpu}, "
            f"vram_gb={DEVICE_INFO.get('vram_gb')}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}) ..."
        )
        if force_cpu:
            print(">>> [说明] CPU 模式：加载 Qwen3（磁盘卸载），失败将直接报错。")
        _runtime = T2MRuntime(
            config_path=config_yml,
            ckpt_name=ckpt,
            force_cpu=force_cpu,
            disable_prompt_engineering=True,
        )
        steps = hm.get("validation_steps")
        if steps is not None:
            for p in _runtime.pipelines:
                p.validation_steps = int(steps)
        for i, p in enumerate(_runtime.pipelines):
            te = getattr(p, "text_encoder", None)
            if te is None or not getattr(te, "has_llm", False):
                raise RuntimeError(
                    f"Pipeline[{i}] 未加载 Qwen 文本编码器。"
                )
            print(f">>> Pipeline[{i}] text_encoder=CLIP+Qwen")
        return _runtime


def _set_stage(job: JobState, stage_id: int, status: str, message: str = ""):
    job.stage = stage_id
    job.stage_name = STAGE_DEFS[stage_id]["name"] if 0 <= stage_id < len(STAGE_DEFS) else ""
    for s in job.stages:
        if s["id"] < stage_id and s["status"] != "error":
            s["status"] = "done"
        elif s["id"] == stage_id:
            s["status"] = status
    job.message = message or job.stage_name
    job.publish()


def _fail(job: JobState, stage_id: int, err: BaseException):
    job.status = "error"
    job.error = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    job.message = f"阶段 {stage_id} 失败: {err}"
    for s in job.stages:
        if s["id"] == stage_id:
            s["status"] = "error"
        elif s["status"] == "running":
            s["status"] = "error"
    job.publish()


def _latest_file(directory: str, suffixes) -> Optional[str]:
    files = rp.list_files(directory, suffixes)
    return files[-1] if files else None


def _prepare_job_dirs(job_id: str) -> Dict[str, str]:
    cfg = rp.load_config(CONFIG_PATH)
    base = rp.resolve(cfg.get("work_dir", "smpl_silverwolf_pipeline/smpl_silverwolf_pipeline/output_pipeline"))
    work = osp.join(base, "web_jobs", job_id)
    dirs = cfg.get("dirs", {})
    paths = {
        "work": work,
        "npz": osp.join(work, dirs.get("npz", "00_npz")),
        "smpl_fbx": osp.join(work, dirs.get("smpl_fbx", "01_smpl_fbx")),
        "retarget": osp.join(work, dirs.get("retarget_fbx", "02_silverwolf_fbx")),
        "binary": osp.join(work, dirs.get("binary_fbx", "03_silverwolf_bin")),
        "vmd": osp.join(work, dirs.get("vmd", "04_vmd")),
        "video": osp.join(work, "05_video"),
    }
    for d in paths.values():
        os.makedirs(d, exist_ok=True)
    return paths


def _run_stage0(job: JobState, text: str, frames: int, npz_dir: str, cfg: dict) -> str:
    """返回生成的 npz 路径。"""
    _set_stage(job, 0, "running", "加载模型并推理（CLIP + Qwen）")
    hm = cfg.get("hymotion", {}) or {}
    runtime = get_runtime()
    job.infer_total = int(getattr(runtime.pipelines[0], "validation_steps", 20) or 20)
    job.infer_step = 0
    job.publish()

    def on_step(step: int, total: int):
        job.infer_step = step
        job.infer_total = total
        job.message = f"推理中 {step}/{total}"
        job.publish()

    duration = max(frames, 1) / 30.0
    stem = "motion"
    # 清空旧产物，避免阶段1吃到历史文件
    for name in os.listdir(npz_dir):
        p = osp.join(npz_dir, name)
        if osp.isfile(p):
            os.remove(p)

    runtime.generate_motion(
        text=text,
        seeds_csv=str(random.randint(0, 999)),
        duration=duration,
        cfg_scale=float(hm.get("cfg_scale", 5.0)),
        output_format="dict",  # 只要 npz，阶段1用模板 FBX
        output_dir=npz_dir,
        output_filename=stem,
        original_text=text,
        progress_callback=on_step,
    )
    npz = _latest_file(npz_dir, (".npz",))
    if not npz:
        raise RuntimeError("阶段0未生成 .npz")
    _set_stage(job, 0, "done", "推理完成")
    return npz


def _run_cmd_capture(cmd: List[str], cwd: Optional[str] = None) -> None:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"$ {printable}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"命令失败 (exit={result.returncode}): {printable}\n{err}")


def _run_stages_1_to_4(job: JobState, cfg: dict, paths: dict) -> str:
    """返回 vmd 路径。"""
    # 临时把 work 子目录接到 cfg 期望的结构：直接调用 stage 函数
    _set_stage(job, 1, "running")
    rp.stage1_npy_to_fbx(cfg, paths["npz"], paths["smpl_fbx"])
    _set_stage(job, 1, "done")

    _set_stage(job, 2, "running")
    rp.stage2_retarget(cfg, paths["smpl_fbx"], paths["retarget"])
    _set_stage(job, 2, "done")

    _set_stage(job, 3, "running")
    rp.stage3_ascii_to_binary(cfg, paths["retarget"], paths["binary"])
    _set_stage(job, 3, "done")

    _set_stage(job, 4, "running")
    rp.stage4_fbx_to_vmd(cfg, paths["binary"], paths["vmd"])
    vmd = _latest_file(paths["vmd"], (".vmd",))
    if not vmd or not osp.isfile(vmd):
        raise RuntimeError("阶段4未生成 .vmd（请确认 Blender 已安装 mmd_tools）")
    _set_stage(job, 4, "done")
    return vmd


def _frames_to_mp4(frame_dir: str, out_mp4: str, fps: int = 30) -> None:
    import glob
    import re

    frames = sorted(glob.glob(osp.join(frame_dir, "frame_*.png")))
    if not frames:
        raise RuntimeError(f"渲染目录无 PNG 帧: {frame_dir}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        sample = osp.basename(frames[0])
        m = re.match(r"frame_(\d+)\.png$", sample)
        digits = len(m.group(1)) if m else 4
        pattern = osp.join(frame_dir, f"frame_%0{digits}d.png")
        cmd = [
            ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and osp.isfile(out_mp4):
            return

    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise RuntimeError(
            "无法合成 MP4：请安装 ffmpeg 或执行 pip install imageio imageio-ffmpeg"
        ) from e

    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8)
    try:
        for p in frames:
            writer.append_data(imageio.imread(p))
    finally:
        writer.close()
    if not osp.isfile(out_mp4):
        raise RuntimeError("imageio 未能写出 MP4")


def _render_video(job: JobState, cfg: dict, pmx: str, vmd: str, video_dir: str) -> str:
    _set_stage(job, 5, "running", "Blender 渲染视频中（较慢）…")
    blender = rp.require_exe(cfg["executables"].get("blender"), "blender", "5")
    out_mp4 = osp.join(video_dir, osp.splitext(osp.basename(vmd))[0] + ".mp4")
    frame_dir = osp.join(video_dir, "frames")
    if osp.isdir(frame_dir):
        shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)
    if osp.isfile(out_mp4):
        os.remove(out_mp4)
    cmd = [blender, "-b", "--python", RENDER_SCRIPT, "--", pmx, vmd, frame_dir]
    _run_cmd_capture(cmd, cwd=REPO_ROOT)
    job.message = "合成 MP4…"
    job.publish()
    _frames_to_mp4(frame_dir, out_mp4, fps=30)
    shutil.rmtree(frame_dir, ignore_errors=True)
    _set_stage(job, 5, "done", "视频渲染完成")
    return out_mp4


def _launch_mmd(cfg: dict, pmx: str, vmd: str) -> None:
    """启动 MMD；标准版无 CLI 导入参数，尽量把工作目录设到模型旁并打开程序。"""
    mmd = (cfg.get("executables") or {}).get("mmd") or ""
    if not mmd or not osp.isfile(mmd):
        # 常见安装路径回退
        candidates = [
            r"C:\Program Files\MikuMikuDanceE_v932x64\MikuMikuDance.exe",
            r"C:\Users\22927\Desktop\MMD932\MikuMikudance.exe",
        ]
        mmd = next((c for c in candidates if osp.isfile(c)), "")
    if not mmd:
        print(">>> [WARN] 未找到 MikuMikuDance.exe，跳过启动 MMD")
        return
    # 写一份简短说明到同目录，方便手动导入
    note = osp.join(osp.dirname(vmd), "OPEN_IN_MMD.txt")
    with open(note, "w", encoding="utf-8") as f:
        f.write("请在 MMD 中：\n")
        f.write(f"1. 文件 → 加载模型\n   {pmx}\n")
        f.write(f"2. 文件 → 加载动作\n   {vmd}\n")
    try:
        subprocess.Popen([mmd], cwd=osp.dirname(mmd))
        print(f">>> 已启动 MMD: {mmd}")
    except Exception as e:
        print(f">>> [WARN] 启动 MMD 失败: {e}")


def _execute_pipeline(job: JobState, text: str, frames: int, character: str = "silver_wolf_lv999") -> None:
    """实际执行 pipeline（可在主进程或 worker 子进程中调用）。"""
    job.character = character
    job.status = "running"
    job.message = f"使用角色：{next((c['label'] for c in CHARACTERS if c['id'] == character), character)}"
    job.publish()
    cfg = rp.load_config(CONFIG_PATH)
    cfg.setdefault("executables", {})
    paths = _prepare_job_dirs(job.job_id)
    pmx = rp.resolve((cfg.get("vmd") or {}).get("pmx_model"))
    if not pmx or not osp.isfile(pmx):
        raise FileNotFoundError(f"找不到银狼 PMX: {pmx}")
    job.pmx_path = pmx

    _run_stage0(job, text.strip(), frames, paths["npz"], cfg)
    vmd = _run_stages_1_to_4(job, cfg, paths)
    job.vmd_path = vmd
    video = _render_video(job, cfg, pmx, vmd, paths["video"])

    static_dir = osp.join(REPO_ROOT, "output", "web_videos")
    os.makedirs(static_dir, exist_ok=True)
    static_name = f"{job.job_id}.mp4"
    static_path = osp.join(static_dir, static_name)
    shutil.copy2(video, static_path)
    job.video_url = f"/videos/{static_name}"

    _launch_mmd(cfg, pmx, vmd)

    job.status = "done"
    job.message = "全部完成"
    for s in job.stages:
        if s["status"] != "error":
            s["status"] = "done"
    job.publish()


def _apply_worker_snapshot(job: JobState, snap: dict) -> None:
    for key in (
        "status", "stage", "stage_name", "stages", "infer_step", "infer_total",
        "message", "error", "video_url", "vmd_path", "pmx_path", "character",
    ):
        if key in snap and snap[key] is not None:
            setattr(job, key, snap[key])
    job.publish()


def run_job(job_id: str, text: str, frames: int, character: str = "silver_wolf_lv999"):
    """在子进程中跑重任务，避免加载 Qwen 时 OOM 把 Flask 一并杀掉。"""
    job = JOBS[job_id]
    job.character = character
    if not _run_lock.acquire(blocking=False):
        job.status = "error"
        job.error = "已有任务在运行，请等待完成后再试。"
        job.message = job.error
        job.publish()
        return

    worker = osp.join(REPO_ROOT, "scripts", "web_job_worker.py")
    cmd = [
        sys.executable, "-u", worker,
        "--job-id", job_id,
        "--text", text,
        "--frames", str(frames),
        "--character", character,
    ]
    env = os.environ.copy()
    env["HY_MOTION_DEVICE"] = os.environ.get("HY_MOTION_DEVICE", DEVICE_INFO.get("device", "cpu"))
    if DEVICE_INFO.get("use_cpu"):
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["HIP_VISIBLE_DEVICES"] = "-1"
    env.setdefault("USE_HF_MODELS", "0")
    # Windows 管道默认 GBK，强制 UTF-8 避免前端进度乱码
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    job.status = "running"
    job.message = "启动工作进程…"
    job.publish()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("{"):
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    print(line, flush=True)
                    continue
                mtype = msg.get("type")
                if mtype in ("snapshot", "done", "error"):
                    _apply_worker_snapshot(job, msg)
                    if mtype == "error" and msg.get("error"):
                        job.error = msg["error"]
                        job.status = "error"
                        job.publish()
                else:
                    print(line, flush=True)
            else:
                # 透传 worker 日志到终端
                print(line, flush=True)

        rc = proc.wait()
        if job.status == "done":
            return
        if rc != 0 or job.status != "done":
            # 无 Python 异常时进程消失：典型是 Windows OOM 杀进程
            if not job.error:
                ram = DEVICE_INFO.get("ram_gb")
                ram_tip = f"本机内存约 {ram}GB，" if ram else ""
                job.error = (
                    f"工作进程异常退出（exit={rc}）。\n"
                    f"{ram_tip}加载 Qwen3-8B 文本编码器约需 15GB+ 内存；"
                    "内存不足时 Windows 会直接结束进程（不会留下 Python 报错）。\n"
                    "已启用磁盘卸载以降低峰值内存。请关闭其它占内存程序后重试；"
                    "若仍失败，需要更大内存或 ≥30GB 显存的 GPU。"
                )
            job.status = "error"
            job.message = "进程被系统终止（多为内存不足）"
            for s in job.stages:
                if s["status"] == "running" or (s["id"] == max(job.stage, 0) and s["status"] != "done"):
                    s["status"] = "error"
            job.publish()
    except Exception as e:
        stage_id = job.stage if job.stage >= 0 else 0
        _fail(job, stage_id, e)
    finally:
        _run_lock.release()


# --------------------------------------------------------------------------- #
#  Web (Flask)
# --------------------------------------------------------------------------- #
def create_app():
    try:
        from flask import Flask, Response, jsonify, request, send_from_directory
    except ImportError as e:
        raise SystemExit(
            "需要安装 flask：\n  conda run -n hymotion-mmd python -m pip install flask\n" + str(e)
        )

    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return INDEX_HTML

    @app.get("/api/info")
    def api_info():
        return jsonify({
            "ok": True,
            "device": DEVICE_INFO,
            "characters": CHARACTERS,
        })

    @app.get("/videos/<path:name>")
    def videos(name: str):
        return send_from_directory(osp.join(REPO_ROOT, "output", "web_videos"), name)

    @app.post("/api/run")
    def api_run():
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        frames = int(data.get("frames") or 90)
        frames = max(30, min(frames, 360))
        character = (data.get("character") or "silver_wolf_lv999").strip()
        valid_ids = {c["id"] for c in CHARACTERS}
        if character not in valid_ids:
            return jsonify({"ok": False, "error": f"未知角色: {character}"}), 400
        if not text:
            return jsonify({"ok": False, "error": "请输入文本描述"}), 400
        job_id = uuid.uuid4().hex[:12]
        job = JobState(job_id=job_id, text=text, character=character)
        with JOBS_LOCK:
            JOBS[job_id] = job
        threading.Thread(target=run_job, args=(job_id, text, frames, character), daemon=True).start()
        return jsonify({"ok": True, "job_id": job_id, "device": DEVICE_INFO})

    @app.get("/api/status/<job_id>")
    def api_status(job_id: str):
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "任务不存在"}), 404
        return jsonify({"ok": True, **job.snapshot()})

    @app.get("/api/events/<job_id>")
    def api_events(job_id: str):
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "任务不存在"}), 404

        def stream():
            # 先推当前快照
            yield f"data: {json.dumps(job.snapshot(), ensure_ascii=False)}\n\n"
            while True:
                try:
                    snap = job.events.get(timeout=1.0)
                    yield f"data: {json.dumps(snap, ensure_ascii=False)}\n\n"
                    if snap.get("status") in ("done", "error"):
                        break
                except queue.Empty:
                    # heartbeat
                    yield f"data: {json.dumps(job.snapshot(), ensure_ascii=False)}\n\n"
                    if job.status in ("done", "error"):
                        break

        return Response(stream(), mimetype="text/event-stream; charset=utf-8", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8",
        })

    return app


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>文本生成MMD动作 DEMO</title>
<style>
  :root {
    --bg: #f5f7fb;
    --card: #ffffff;
    --line: #e5eaf2;
    --text: #1f2937;
    --muted: #6b7280;
    --accent: #2563eb;
    --ok: #16a34a;
    --err: #dc2626;
    --run: #d97706;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text); min-height: 100vh;
  }
  .wrap { max-width: 960px; margin: 0 auto; padding: 32px 20px 60px; }
  h1 { font-size: 1.6rem; margin: 0 0 6px; font-weight: 650; color: #111827; }
  .sub { color: var(--muted); margin-bottom: 24px; line-height: 1.5; }
  .card {
    background: var(--card);
    border: 1px solid var(--line); border-radius: 14px; padding: 20px;
    margin-bottom: 18px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
  }
  label { display:block; font-size: .9rem; color: var(--muted); margin-bottom: 8px; }
  textarea, input[type=number] {
    width: 100%; background: #fff; color: var(--text);
    border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px;
    font-size: 1rem; outline: none;
  }
  textarea:focus, input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37,99,235,.12); }
  textarea { min-height: 110px; resize: vertical; }
  .row { display:flex; gap: 14px; flex-wrap: wrap; align-items: end; }
  .row .field { flex: 1; min-width: 140px; }
  button {
    background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; border: 0;
    border-radius: 10px; padding: 12px 22px; font-size: 1rem; cursor: pointer;
    font-weight: 600; min-width: 140px;
  }
  button:disabled { opacity: .5; cursor: not-allowed; }
  button:hover:not(:disabled) { filter: brightness(1.05); }
  .stages { display: grid; gap: 10px; min-height: 0; }
  .stages:empty::before {
    content: "尚未开始";
    color: var(--muted);
    font-size: .9rem;
  }
  .stage {
    display:flex; align-items:center; gap: 12px; padding: 10px 12px;
    border-radius: 10px; background: #f8fafc; border: 1px solid var(--line);
  }
  .dot {
    width: 12px; height: 12px; border-radius: 50%; background: #cbd5e1;
    flex: 0 0 auto;
  }
  .stage.running .dot { background: var(--run); box-shadow: 0 0 0 4px rgba(217,119,6,.15); }
  .stage.done .dot { background: var(--ok); }
  .stage.error .dot { background: var(--err); }
  .stage-name { flex: 1; }
  .stage-status { color: var(--muted); font-size: .85rem; }
  #inferBox { display: none; margin-top: 16px; }
  .bar-wrap { height: 12px; background: #eef2f7; border-radius: 999px; overflow: hidden; border: 1px solid var(--line); }
  .bar { height: 100%; width: 0%; background: linear-gradient(90deg, #3b82f6, #60a5fa); transition: width .2s; }
  .msg { color: var(--muted); margin-top: 10px; min-height: 1.2em; }
  .error {
    white-space: pre-wrap; background: #fef2f2; color: #991b1b;
    border: 1px solid #fecaca; border-radius: 10px; padding: 12px; display:none;
    font-family: ui-monospace, Consolas, monospace; font-size: .85rem; max-height: 280px; overflow: auto;
  }
  video { width: 100%; border-radius: 12px; background: #111; margin-top: 8px; }
  .paths { font-size: .85rem; color: var(--muted); word-break: break-all; }
  .notice {
    display: none; margin-bottom: 18px; padding: 12px 14px; border-radius: 10px;
    border: 1px solid #fde68a; background: #fffbeb; color: #92400e; font-size: .95rem;
  }
  .notice.gpu {
    border-color: #bbf7d0; background: #f0fdf4; color: #166534;
  }
  select {
    width: 100%; background: #fff; color: var(--text);
    border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px;
    font-size: 1rem; outline: none;
  }
  select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37,99,235,.12); }
</style>
</head>
<body>
<div class="wrap">
  <h1>文本生成MMD动作 DEMO</h1>
  <p class="sub">输入动作描述，自动执行：文本生成 → SMPL FBX → 银狼重定向 → VMD → 渲染视频。完成后启动 MMD，并在下方播放渲染结果。</p>

  <div id="deviceNotice" class="notice"></div>

  <div class="card">
    <label for="character">选择角色</label>
    <select id="character">
      <option value="silver_wolf_lv999">银狼 LV.999</option>
    </select>
    <label for="text" style="margin-top:14px">动作描述（英文效果通常更好）</label>
    <textarea id="text" placeholder="A person walks forward slowly.">A person walks forward slowly.</textarea>
    <div class="row" style="margin-top:14px">
      <div class="field">
        <label for="frames">动作长度（帧，30fps；会直接决定生成时长，非模型自动估长）</label>
        <input id="frames" type="number" min="30" max="360" value="90"/>
      </div>
      <button id="btn" onclick="startJob()">开始生成</button>
    </div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
      <strong>流水线进度</strong>
      <span id="overall" class="stage-status">等待开始</span>
    </div>
    <div class="stages" id="stages"></div>
    <div id="inferBox">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span class="stage-status">推理进度</span>
        <span class="stage-status" id="inferLabel">-</span>
      </div>
      <div class="bar-wrap"><div class="bar" id="inferBar"></div></div>
    </div>
    <p class="msg" id="msg"></p>
    <pre class="error" id="error"></pre>
  </div>

  <div class="card">
    <strong>渲染视频</strong>
    <video id="video" controls playsinline></video>
    <p class="paths" id="paths"></p>
  </div>
</div>
<script>
const STAGE_NAMES = [
  "HY-Motion 文本生成动作",
  "SMPL FBX 转换",
  "银狼动作重定向",
  "ASCII → 二进制 FBX",
  "导出 VMD",
  "渲染视频 (PMX+VMD)",
];
const stagesEl = document.getElementById("stages");
const statusText = {pending:"等待", running:"进行中", done:"完成", error:"失败"};
let es = null;
let deviceInfo = null;

async function loadInfo() {
  try {
    const resp = await fetch("/api/info");
    const data = await resp.json();
    if (!data.ok) return;
    deviceInfo = data.device || {};
    const noticeEl = document.getElementById("deviceNotice");
    if (deviceInfo.notice) {
      noticeEl.style.display = "block";
      noticeEl.textContent = deviceInfo.notice;
      noticeEl.className = "notice" + (deviceInfo.use_cpu ? "" : " gpu");
    }
    const sel = document.getElementById("character");
    sel.innerHTML = "";
    (data.characters || []).forEach(c => {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = c.label;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error(e);
  }
}
loadInfo();

function renderStages(stages) {
  stagesEl.innerHTML = "";
  (stages || []).forEach(st => {
    // 只显示已完成 / 进行中 / 失败的步骤，未开始的不展示
    if (!st.status || st.status === "pending") return;
    const div = document.createElement("div");
    div.className = "stage " + st.status;
    const name = STAGE_NAMES[st.id] || st.name || ("步骤 " + st.id);
    div.innerHTML =
      `<div class="dot"></div>` +
      `<div class="stage-name">${st.id}. ${name}</div>` +
      `<div class="stage-status">${statusText[st.status] || st.status}</div>`;
    stagesEl.appendChild(div);
  });
}

function applyState(s) {
  document.getElementById("msg").textContent = s.message || "";
  document.getElementById("overall").textContent =
    s.status === "done" ? "全部完成" :
    s.status === "error" ? "出错" :
    s.status === "running" ? `进行中：步骤 ${s.stage}` : "排队中";

  renderStages(s.stages);

  const inferBox = document.getElementById("inferBox");
  const showInfer = s.stage === 0 && (s.status === "running" || (s.infer_total > 0 && s.infer_step < s.infer_total));
  const stage0 = (s.stages || []).find(x => x.id === 0);
  if (stage0 && (stage0.status === "running" || (stage0.status === "done" && s.stage === 0))) {
    inferBox.style.display = "block";
  } else if (stage0 && stage0.status === "done" && s.stage > 0) {
    inferBox.style.display = "none";
  } else if (showInfer) {
    inferBox.style.display = "block";
  } else if (!s.status || s.status === "queued") {
    inferBox.style.display = "none";
  }

  const total = s.infer_total || 0;
  const step = s.infer_step || 0;
  const pct = total > 0 ? Math.min(100, Math.round(step / total * 100)) : (s.stage > 0 ? 100 : 0);
  document.getElementById("inferBar").style.width = pct + "%";
  document.getElementById("inferLabel").textContent = total > 0 ? `${step} / ${total} (${pct}%)` : "-";

  const err = document.getElementById("error");
  if (s.error) {
    err.style.display = "block";
    err.textContent = s.error;
  } else {
    err.style.display = "none";
    err.textContent = "";
  }

  if (s.video_url) {
    const v = document.getElementById("video");
    if (v.src.indexOf(s.video_url) < 0) {
      v.src = s.video_url + "?t=" + Date.now();
      v.play().catch(()=>{});
    }
  }
  const paths = [];
  if (s.pmx_path) paths.push("PMX: " + s.pmx_path);
  if (s.vmd_path) paths.push("VMD: " + s.vmd_path);
  document.getElementById("paths").textContent = paths.join("\n");

  if (s.status === "done" || s.status === "error") {
    document.getElementById("btn").disabled = false;
    if (es) { es.close(); es = null; }
  }
}

async function startJob() {
  const text = document.getElementById("text").value.trim();
  const frames = parseInt(document.getElementById("frames").value || "90", 10);
  const character = document.getElementById("character").value;
  if (!text) { alert("请输入文本"); return; }
  document.getElementById("btn").disabled = true;
  document.getElementById("error").style.display = "none";
  document.getElementById("video").removeAttribute("src");
  stagesEl.innerHTML = "";
  document.getElementById("inferBox").style.display = "none";
  document.getElementById("inferBar").style.width = "0%";
  document.getElementById("overall").textContent = "排队中";
  document.getElementById("msg").textContent = "";

  const resp = await fetch("/api/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text, frames, character}),
  });
  const data = await resp.json();
  if (!data.ok) {
    alert(data.error || "启动失败");
    document.getElementById("btn").disabled = false;
    return;
  }
  if (es) es.close();
  es = new EventSource("/api/events/" + data.job_id);
  es.onmessage = (ev) => {
    try { applyState(JSON.parse(ev.data)); } catch (e) { console.error(e); }
  };
  es.onerror = () => {};
}
</script>
</body>
</html>
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="文本生成MMD动作 DEMO")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app = create_app()
    url = f"http://{args.host}:{args.port}/"
    print("=" * 60)
    print("文本生成MMD动作 DEMO")
    print(f"  {url}")
    print(f"  device={DEVICE_INFO.get('device')}  vram_gb={DEVICE_INFO.get('vram_gb')}  gpu={DEVICE_INFO.get('gpu_name')}")
    print(f"  notice={DEVICE_INFO.get('notice')}")
    print(f"  HY_MOTION_DEVICE={os.environ.get('HY_MOTION_DEVICE')}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    print(f"  USE_HF_MODELS={os.environ.get('USE_HF_MODELS')}")
    print("=" * 60)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
