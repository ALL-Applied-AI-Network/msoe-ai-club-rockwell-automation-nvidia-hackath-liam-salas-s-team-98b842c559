"""Floor-free inference of dense53 over all five real cap_seal videos: every component (mean probability >= 0.5,
>= 9 px) with its box and confidence, one JSON line per frame, so confidence rules can be scored offline against
the frozen scorer without re-running the GPU. Usage: raw_infer.py OUT_DIR"""
import json, sys, time, cv2, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
N = Path('/home/hackathon5/outputs/topdown-01a17f15-20261029'); out = Path(sys.argv[1]); out.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
from fluidmix.topdown.segmenter import DenseCapNet, image_tensor, instances
fm = json.load(open(N / 'dense53/train01/frozen-model.json'))
m = DenseCapNet(); m.load_state_dict(torch.load(fm['weights'], map_location='cpu', weights_only=True)); m.cuda().eval()
man = json.load(open(N / 'dense-real53/manifest.json'))
V = Path('/home/hackathon5/liamreporoot/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/external/external/csi-vial-fill-dataset')
for rec in man['records']:
    if rec['camera'] != 'cap_seal': continue
    stem = Path(rec['path']).stem; target = out / f'{stem}.jsonl'
    if target.exists(): print('skip', stem); continue
    cap = cv2.VideoCapture(str(V / rec['path'])); n = 0; t0 = time.time()
    with open(str(target) + '.part', 'w') as fh:
        while n < rec['frames']:
            images = []
            for _ in range(min(16, rec['frames'] - n)):
                ok, im = cap.read()
                if not ok: raise SystemExit(f'decode failed {stem} {n}')
                images.append(im)
            with torch.no_grad(): probs = m(image_tensor(images, 'cuda')).sigmoid().cpu().numpy()[:, 0]
            for p in probs:
                boxes, scores, _ = instances(p)
                fh.write(json.dumps(dict(frame=n, boxes_xyxy=boxes, confidence=[round(float(s), 6) for s in scores])) + '\n'); n += 1
            if n % 8192 == 0: print(stem, n, f'{n/(time.time()-t0):.0f} fps', flush=True)
    cap.release(); Path(str(target) + '.part').rename(target); print('done', stem, n, f'{time.time()-t0:.0f}s', flush=True)
