import os

os.environ["USE_HF_MODELS"] = "0"

from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer

print("Loading CLIP...")
CLIPTokenizer.from_pretrained("ckpts/clip-vit-large-patch14")
CLIPTextModel.from_pretrained("ckpts/clip-vit-large-patch14")
print("CLIP OK")

print("Loading Qwen tokenizer...")
AutoTokenizer.from_pretrained("ckpts/Qwen3-8B")
print("Qwen tokenizer OK")
print("All text encoders verified.")
