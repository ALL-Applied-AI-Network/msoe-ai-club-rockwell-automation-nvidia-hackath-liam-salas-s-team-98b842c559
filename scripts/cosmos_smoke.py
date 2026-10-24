"""Clean Cosmos3 reasoner smoke: load a pinned HF-cache snapshot and describe one image.

    hf download nvidia/Cosmos3-Edge --revision <sha> --include '*.json' --include '*.jinja' --include '*.txt' \
        --include '*.md' --include 'transformer/*' --include 'vision_encoder/*' --include 'text_tokenizer/*'
    .venv/bin/python cosmos_smoke.py --model nvidia/Cosmos3-Edge --revision <sha> --image frame.png --out result.json

Runtime/compatibility check only: a non-empty, on-topic completion says the stack works, not that the
model detects defects or explains them correctly.
"""
import argparse, hashlib, json, platform, time

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--revision", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--prompt", default="Describe only what is visibly present in this synthetic vial-fill camera "
                "frame in one short sentence. Do not infer defects, causes, recipe history, or operator actions.")
ap.add_argument("--max-new-tokens", type=int, default=96)
ap.add_argument("--no-thinking", action="store_true",
                help="pass enable_thinking=False to chat templates that support it (Cosmos3-Edge does)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

r = {"model": a.model, "revision": a.revision, "image": a.image, "prompt": a.prompt,
     "image_sha256": hashlib.sha256(open(a.image, "rb").read()).hexdigest(),
     "image_size": Image.open(a.image).size, "torch": torch.__version__, "platform": platform.platform(),
     "device": torch.cuda.get_device_name(0)}
# Reasoner-only snapshot: the generator's vae/scheduler/sound files are deliberately not downloaded.
REASONER = ["*.json", "*.jinja", "*.txt", "*.md", "transformer/*", "vision_encoder/*", "text_tokenizer/*"]
path = snapshot_download(a.model, revision=a.revision, allow_patterns=REASONER, local_files_only=True)

t = time.monotonic()
processor = AutoProcessor.from_pretrained(path)
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
torch.cuda.synchronize(); r["load_seconds"] = round(time.monotonic() - t, 1)
r["params_billion"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)

messages = [{"role": "user", "content": [{"type": "image", "path": a.image}, {"type": "text", "text": a.prompt}]}]
extra = {"enable_thinking": False} if a.no_thinking else {}
r["enable_thinking"] = not a.no_thinking
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt", **extra).to(model.device, torch.bfloat16)
t = time.monotonic()
with torch.inference_mode():
    out = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_new_tokens)
torch.cuda.synchronize(); r["generation_seconds"] = round(time.monotonic() - t, 2)
new = out[0, inputs.input_ids.shape[-1]:]
r["generated_tokens"] = int(new.shape[-1])
r["completion"] = processor.decode(new, skip_special_tokens=True)
r["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
r["status"] = "pass" if r["completion"].strip() else "fail"
json.dump(r, open(a.out, "w"), indent=1)
print(json.dumps(r, indent=1))
