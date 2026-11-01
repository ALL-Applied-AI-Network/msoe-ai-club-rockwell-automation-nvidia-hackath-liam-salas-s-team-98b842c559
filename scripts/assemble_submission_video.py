#!/usr/bin/env python
"""Assemble a captioned draft of the five-minute submission video from the demo clips, figures and the script.

No audio: the narration from docs/submission/video-script.md is burned in as subtitles, block by block, so the draft
can be watched silently or voiced over later. Output is 1280x720 at 30 fps in VP9 WebM (what OpenCV can encode here;
YouTube and most players accept it). Inputs are named below; missing clips fall back to a title card so the draft
always builds. Usage: assemble_submission_video.py [--out PATH]
"""
import argparse, re, textwrap, cv2, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SPARK = Path('/home/hackathon5/outputs/topdown-01a17f15-20261029/iteration-20261031-liam/demo')
CARRY = Path('/home/hackathon5/liamreporoot/topdown-results-20261101/outputs/carryover-video5/demo')
DATA = Path('/home/hackathon5/liamreporoot/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/external/external/csi-vial-fill-dataset')
FIG = ROOT / 'docs/assets/topdown'; FIGC = ROOT / 'docs/assets/carryover'
W, H, FPS = 1280, 720, 30
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_script(path):
    """Return [(start_s, end_s, narration)] from the script's bold timestamp headers and quoted blocks."""
    text = path.read_text(); blocks = []
    for m in re.finditer(r"\*\*(\d+):(\d\d) to (\d+):(\d\d), ([^.]*)\.\*\*(.*?)(?=\n\*\*\d+:\d\d to|\n---)", text, re.S):
        s = int(m.group(1)) * 60 + int(m.group(2)); e = int(m.group(3)) * 60 + int(m.group(4))
        narration = ' '.join(l[2:].strip() for l in m.group(6).splitlines() if l.startswith('> '))
        blocks.append((s, e, m.group(5), narration))
    return blocks


def fit(img, w=W, h=H):
    """Letterbox an image into w x h."""
    ih, iw = img.shape[:2]; s = min(w / iw, h / ih); nw, nh = int(iw * s), int(ih * s)
    canvas = np.zeros((h, w, 3), np.uint8); r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (h - nh) // 2, (w - nw) // 2; canvas[y:y + nh, x:x + nw] = r; return canvas


def card(lines, sub=None):
    img = np.full((H, W, 3), 18, np.uint8); y = H // 2 - 30 * len(lines)
    for i, l in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(l, FONT, 1.1, 2); cv2.putText(img, l, ((W - tw) // 2, y + i * 64), FONT, 1.1, (235, 235, 235), 2, cv2.LINE_AA)
    if sub: cv2.putText(img, sub, (40, H - 60), FONT, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
    return img


def video_frames(path, start=0, count=None, step=1):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, start); n = 0
    while count is None or n < count:
        ok, f = cap.read()
        if not ok: break
        if n % step == 0: yield f
        n += 1
    cap.release()


def results_table():
    img = np.full((H, W, 3), 18, np.uint8)
    rows = [('model', 'training data', 'precision', 'recall', 'identity coverage'),
            ('team YOLO baseline', 'twin renders only', '0.945', '0.873', '0.78'),
            ('ours, DenseCapNet', 'twin renders only', '0.957', '0.920', '0.65'),
            ('ours, reproducible variant', 'twin renders only', '0.998', '0.851', '0.44')]
    cv2.putText(img, 'Five real cap_seal videos, 105k frames, frozen harness', (60, 110), FONT, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    xs = [60, 420, 760, 920, 1060]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            col = (160, 160, 160) if r == 0 else ((120, 230, 160) if r == 2 else (235, 235, 235))
            cv2.putText(img, cell, (xs[c], 200 + r * 70), FONT, 0.7, col, 2 if r == 2 else 1, cv2.LINE_AA)
    cv2.putText(img, 'ahead on per-frame detection, behind on track identity; exposed validation, not a blind test', (60, 560), FONT, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
    return img


def shots(block_index, seconds):
    """Yield frames (W x H) for a block; each shot a (source, seconds) pair split evenly across the block."""
    n = int(seconds * FPS)
    plan = {
        0: [('vid', DATA / 'cap_seal/videos/cap_seal_video_5.mp4', 43100), ('vid', DATA / 'tank_level/videos/tank_level_video_5.mp4', 43100)],
        1: [('card', ['Fail-closed evidence pipeline', 'top-down identity + dispense timing + vial colour', 'keep / discard / insufficient evidence'], None), ('img', FIG / 'real_vs_synth.jpg', None)],
        2: [('vid', SPARK / 'demo_v4.mp4', 0), ('tbl', None, None)],
        3: [('img', FIG / 'real41_lane.jpg', None), ('img', FIG / 'aug53.jpg', None), ('img', FIG / 'fp48_band.jpg', None)],
        4: [('vid', DATA / 'vial_fill/videos/vial_fill_video_5.mp4', 1400)],
        5: [('vid', CARRY / 'changeover_vial22_recipe11to12_discard.mp4', 0), ('vid', CARRY / 'lanepass_vial25_discard.mp4', 900), ('img', FIGC / 'attribution.png', None)],
        6: [('img', FIG / 'dropout53.jpg', None), ('img', FIG / 'fp57_band8.jpg', None)],
        7: [('card', ['1  No line footage needed: train on the twin, verify on the line', '2  Carryover is a sequencing problem: vial identity names the flush', '3  Fail closed: every flag carries its evidence or abstains'], None), ('img', FIG / 'real_vs_synth.jpg', None)],
        8: [('card', ['Team Mixture Of Experts', 'code, protocols, receipts, review decisions and the carryover run are in the repository', 'weights, scorer, labels and clips on the team DGX Spark, named by path and hash'], 'github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559')],
    }.get(block_index, [('card', ['(shot missing)'], None)])
    per = n // len(plan); made = 0
    for kind, src, arg in plan:
        want = per if made + per <= n else n - made
        if kind == 'vid' and src.exists():
            k = 0
            for f in video_frames(src, arg or 0, want): yield fit(f); k += 1
            for _ in range(want - k): yield fit(f) if k else card(['(clip ended)'])
        elif kind == 'img' and src is not None and src.exists():
            im = fit(cv2.imread(str(src)))
            for _ in range(want): yield im
        elif kind == 'tbl':
            im = results_table()
            for _ in range(want): yield im
        else:
            im = card(src if kind == 'card' else [f'(missing {src})'], arg)
            for _ in range(want): yield im
        made += want
    for _ in range(n - made): yield card(['...'])


def subtitle(frame, text):
    lines = textwrap.wrap(text, 70)[:3]; y0 = H - 24 - 34 * len(lines)
    cv2.rectangle(frame, (0, y0 - 34), (W, H), (0, 0, 0), -1)
    for i, l in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(l, FONT, 0.75, 2); cv2.putText(frame, l, ((W - tw) // 2, y0 + i * 34), FONT, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', default=str(ROOT / 'outputs/submission-video-draft.webm')); a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    blocks = parse_script(ROOT / 'docs/submission/video-script.md')
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'VP90'), FPS, (W, H)); total = 0
    for bi, (s, e, title, narration) in enumerate(blocks):
        seconds = e - s; sentences = [x.strip() for x in re.split(r'(?<=[.;:])\s+', narration) if x.strip()]
        words = sum(len(x.split()) for x in sentences) or 1; frames = list(shots(bi, seconds)); n = len(frames)
        cursor = 0
        for sent in sentences:
            share = int(n * len(sent.split()) / words); seg = frames[cursor:cursor + share]
            for f in seg: wr.write(subtitle(f.copy(), sent))
            cursor += share
        for f in frames[cursor:]: wr.write(subtitle(f.copy(), sentences[-1] if sentences else title))
        total += n; print(f'block {bi} {title}: {seconds} s, {len(sentences)} subtitles', flush=True)
    wr.release(); print(f'wrote {a.out}: {total / FPS:.0f} s')


if __name__ == '__main__':
    main()
