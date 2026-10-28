"""Batched Cosmos3 reasoner pass over a folder of frames, with a batch-size sweep and GPU-utilisation sampling.

    .venv/bin/python cosmos_batch.py --model nvidia/Cosmos3-Edge --revision <sha> --frames <dir>/rgb \
        --batch-sizes 1,8,32 --no-thinking --out cosmos-batch/edge

Throughput/runtime evidence only; captions are not graded for correctness here.
"""
import argparse, json, subprocess, threading, time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--revision", required=True)
ap.add_argument("--frames", required=True, help="directory of PNG frames")
ap.add_argument("--batch-sizes", default="1,8,32")
ap.add_argument("--max-new-tokens", type=int, default=64)
ap.add_argument("--prompt", default="Describe only what is visibly present in this synthetic vial-fill camera "
                "frame in one short sentence. Do not infer defects, causes, recipe history, or operator actions.")
ap.add_argument("--no-thinking", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

REASONER = ["*.json", "*.jinja", "*.txt", "*.md", "transformer/*", "vision_encoder/*", "text_tokenizer/*"]
path = snapshot_download(a.model, revision=a.revision, allow_patterns=REASONER, local_files_only=True)
processor = AutoProcessor.from_pretrained(path)
processor.tokenizer.padding_side = "left"  # decoder-only batching
t = time.monotonic()
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda",
                                                    attn_implementation="sdpa").eval()
load_s = round(time.monotonic() - t, 1)
frames = sorted(Path(a.frames).glob("*.png"))
extra = {"enable_thinking": False} if a.no_thinking else {}


class GpuSampler(threading.Thread):
    """Poll nvidia-smi utilisation every 0.5 s while a batch runs."""
    def __init__(self):
        super().__init__(daemon=True); self.samples = []; self.stop = threading.Event()
    def run(self):
        while not self.stop.is_set():
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip().split(",")
            try:
                self.samples.append((float(q[0]), float(q[1])))
            except (ValueError, IndexError):
                pass
            time.sleep(0.5)


def run_batch(paths):
    convs = [[{"role": "user", "content": [{"type": "image", "path": str(p)}, {"type": "text", "text": a.prompt}]}]
             for p in paths]
    inputs = processor.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True, **extra).to(model.device, torch.bfloat16)
    with torch.inference_mode():
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_new_tokens)
    new = ids[:, inputs.input_ids.shape[-1]:]
    return processor.batch_decode(new, skip_special_tokens=True), int((new != processor.tokenizer.pad_token_id).sum())


sweep = []
run_batch(frames[:1])  # warm-up: CUDA context, kernels, allocator
for bs in [int(x) for x in a.batch_sizes.split(",")]:
    torch.cuda.reset_peak_memory_stats(); s = GpuSampler(); s.start(); t = time.monotonic()
    captions, toks = [], 0
    for i in range(0, len(frames), bs):
        c, n = run_batch(frames[i:i + bs]); captions += c; toks += n
    torch.cuda.synchronize(); dt = time.monotonic() - t; s.stop.set(); s.join()
    util = [u for u, _ in s.samples]; pw = [p for _, p in s.samples]
    row = {"batch_size": bs, "frames": len(frames), "seconds": round(dt, 1),
           "frames_per_s": round(len(frames) / dt, 2), "gen_tokens_per_s": round(toks / dt, 1),
           "gpu_util_mean": round(sum(util) / max(len(util), 1), 1), "gpu_util_max": max(util, default=None),
           "power_w_mean": round(sum(pw) / max(len(pw), 1), 1),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    sweep.append(row); print(json.dumps(row), flush=True)
    (out / f"captions_bs{bs}.json").write_text(json.dumps(dict(zip([p.name for p in frames], captions)), indent=1))

summary = {"model": a.model, "revision": a.revision, "load_seconds": load_s, "enable_thinking": not a.no_thinking,
           "max_new_tokens": a.max_new_tokens, "device": torch.cuda.get_device_name(0), "sweep": sweep}
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
