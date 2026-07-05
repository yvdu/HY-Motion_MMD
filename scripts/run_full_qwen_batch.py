"""本地全流程（含 Qwen 文本编码），不启动网页。

用法:
  python scripts/run_full_qwen_batch.py
"""
from __future__ import annotations

import os
import os.path as osp
import random
import shutil
import sys

os.environ["USE_HF_MODELS"] = "0"
os.environ["HY_MOTION_DEVICE"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONUTF8"] = "1"

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
PKG_ROOT = osp.join(REPO_ROOT, "smpl_silverwolf_pipeline", "smpl_silverwolf_pipeline")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PKG_ROOT)

PROMPTS = [
    ("walk_forward_slowly", "A person walks forward slowly."),
    ("walk_forward_slowly_looking_around", "A person walks forward slowly while looking around."),
]

FRAMES = 90
CFG_SCALE = 5.0


def main() -> None:
    import importlib.util

    from hymotion.utils.t2m_runtime import T2MRuntime

    spec = importlib.util.spec_from_file_location(
        "run_pipeline", osp.join(PKG_ROOT, "run_pipeline.py")
    )
    rp = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rp)

    cfg = rp.load_config(osp.join(PKG_ROOT, "pipeline_config.yaml"))
    cfg.setdefault("executables", {})
    work_base = osp.join(REPO_ROOT, "output", "full_qwen_batch")
    os.makedirs(work_base, exist_ok=True)

    model_path = rp.resolve(cfg.get("hymotion", {}).get("model_path", "ckpts/tencent/HY-Motion-1.0"))
    config_yml = osp.join(model_path, "config.yml")
    ckpt = osp.join(model_path, "latest.ckpt")

    print("=" * 60)
    print("Full pipeline with Qwen text encoder (no web UI)")
    print(f"  model: {model_path}")
    print(f"  device: CPU + Qwen disk offload")
    print(f"  prompts: {len(PROMPTS)}")
    print("=" * 60)

    print(">>> Loading T2MRuntime (this may take several minutes)...")
    runtime = T2MRuntime(
        config_path=config_yml,
        ckpt_name=ckpt,
        force_cpu=True,
        disable_prompt_engineering=True,  # 无 Text2MotionPrompter；语义靠 Qwen 编码原始文本
    )
    steps = (cfg.get("hymotion") or {}).get("validation_steps", 20)
    if steps is not None:
        for p in runtime.pipelines:
            p.validation_steps = int(steps)

    te = runtime.pipelines[0].text_encoder
    if not getattr(te, "has_llm", False):
        raise RuntimeError("Qwen failed to load (has_llm=False). Aborting.")
    print(">>> Text encoder: CLIP + Qwen OK")

    duration = FRAMES / 30.0
    results = []

    for stem, text in PROMPTS:
        print("\n" + "=" * 60)
        print(f">>> Prompt: {text}")
        print("=" * 60)

        job_dir = osp.join(work_base, stem)
        dirs = {
            "npz": osp.join(job_dir, "00_npz"),
            "smpl_fbx": osp.join(job_dir, "01_smpl_fbx"),
            "retarget": osp.join(job_dir, "02_silverwolf_fbx"),
            "binary": osp.join(job_dir, "03_silverwolf_bin"),
            "vmd": osp.join(job_dir, "04_vmd"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
            for name in os.listdir(d):
                p = osp.join(d, name)
                if osp.isfile(p):
                    os.remove(p)

        def on_step(step, total, _text=text):
            print(f"\r>>> infer {_text[:40]}... {step}/{total}", end="", flush=True)

        print(">>> Stage 0: HY-Motion generate (Qwen encode original text)")
        runtime.generate_motion(
            text=text,
            seeds_csv=str(random.randint(0, 999)),
            duration=duration,
            cfg_scale=CFG_SCALE,
            output_format="dict",
            output_dir=dirs["npz"],
            output_filename=stem,
            original_text=text,
            progress_callback=on_step,
        )
        print()
        npz_files = rp.list_files(dirs["npz"], (".npz",))
        if not npz_files:
            raise RuntimeError(f"No npz for: {text}")
        print(f">>> npz: {npz_files[-1]}")

        print(">>> Stage 1: SMPL FBX")
        rp.stage1_npy_to_fbx(cfg, dirs["npz"], dirs["smpl_fbx"])
        print(">>> Stage 2: retarget silver_wolf")
        rp.stage2_retarget(cfg, dirs["smpl_fbx"], dirs["retarget"])
        print(">>> Stage 3: ascii -> binary FBX")
        rp.stage3_ascii_to_binary(cfg, dirs["retarget"], dirs["binary"])
        print(">>> Stage 4: FBX -> VMD")
        rp.stage4_fbx_to_vmd(cfg, dirs["binary"], dirs["vmd"])

        vmds = rp.list_files(dirs["vmd"], (".vmd",))
        if not vmds:
            raise RuntimeError(f"No vmd for: {text}")
        src_vmd = vmds[-1]
        out_vmd = osp.join(REPO_ROOT, f"{stem}.vmd")
        shutil.copy2(src_vmd, out_vmd)
        print(f">>> Saved: {out_vmd} ({osp.getsize(out_vmd)} bytes)")
        results.append((text, out_vmd))

    print("\n" + "=" * 60)
    print("DONE")
    for text, path in results:
        print(f"  [{text}]")
        print(f"    -> {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
