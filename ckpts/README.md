# Model Checkpoints Download Guide

This project requires several model weights to function. By default, `app.py` and `gradio_app.py` support automatic downloading from Hugging Face. However, for offline environments or faster startup, we recommend manual downloading.

## 1. Motion Generation Models (T2M Models)

These are the main DiT-based motion generation models. You can choose between the Standard or Lite version.

| Model Name | Hugging Face Repo ID |
| :--- | :--- |
| **HY-Motion-1.0** | `tencent/HY-Motion-1.0/HY-Motion-1.0` |
| **HY-Motion-1.0-Lite** | `tencent/HY-Motion-1.0/HY-Motion-1.0-Lite` |

**Manual Download Command:**
```bash
# Example for Standard version
huggingface-cli download tencent/HY-Motion-1.0 --include "HY-Motion-1.0/*" --local-dir ckpts/tencent

# Example for Lite version
huggingface-cli download tencent/HY-Motion-1.0 --include "HY-Motion-1.0-Lite/*" --local-dir ckpts/tencent
```

## 2. Text Encoders

The model uses CLIP and Qwen as text encoders. According to hymotion/network/text_encoders/text_encoder.py, the paths depend on the USE_HF_MODELS environment variable.

- If USE_HF_MODELS=1 (Default): The code will fetch models directly from Hugging Face using Repo IDs: openai/clip-vit-large-patch14 and Qwen/Qwen3-8B.
- If USE_HF_MODELS=0: The code expects the weights to be located in the ckpts/ directory.

Manual Download Commands:
```bash
# CLIP Large
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ckpts/clip-vit-large-patch14/

# Qwen Text Encoder
huggingface-cli download Qwen/Qwen3-8B --local-dir ckpts/Qwen3-8B
```

## 3. Prompt Rewriter & Duration Predictor (Optional)

If you intend to use the text rewriting and duration estimation module (via vllm as configured in ``start_space.sh``), you will need this model:

Manual Download Command:
```bash
huggingface-cli download Text2MotionPrompter/Text2MotionPrompter --local-dir ckpts/Text2MotionPrompter
```

## Recommended Directory Structure

If you prefer local loading (recommended for stability), your ``ckpts/`` directory should look like this:

```
ckpts/
├── tencent/
│   ├── HY-Motion-1.0/         # Contains config.yml and latest.ckpt
│   └── HY-Motion-1.0-Lite/    # Optional
├── clip-vit-large-patch14/     # CLIP weights
├── Qwen3-8B/                   # Qwen text encoder weights
└── Text2MotionPrompter/        # vLLM Rewriter weights (Optional)
```
Note: Ensure you set ``USE_HF_MODELS=0`` in your environment variables if you want the application to read from the local ``ckpts/`` folder instead of the internet.
