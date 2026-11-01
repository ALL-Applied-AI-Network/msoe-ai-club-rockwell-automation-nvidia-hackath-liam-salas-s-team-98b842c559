"""One bounded, from-scratch dense-cap experiment on verified synthetic frames."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from topdown_dense_segmenter import DenseCapNet, dense_loss, image_tensor, instances
from dense_validation_metrics import frame_metrics, summarize
from instance_balanced_loss import foreground_weights, instance_balanced_loss
from epoch_schedule import epoch_learning_rate, set_epoch_learning_rate

HOLDER = {'scale': [0.8, 1.6], 'holder_probability': 0.5, 'holder_size_factor': [1.4, 2.2], 'holder_value': [15, 75], 'holder_shape': 'square'}
CAPPASTE = {'probability': 0.7, 'copies': [1, 3], 'lane_x': [120, 224], 'source': 'same_frame_instances', 'masks_updated': True}
PROCEDURAL = {'probability': 0.6, 'strokes': [1, 4], 'stroke_thickness': [2, 7], 'stroke_points': [3, 6], 'stroke_span': [30, 120], 'rectangles': [0, 2], 'rectangle_size': [20, 80], 'lane_x': [120, 224], 'stroke_value': [0, 70], 'rectangle_value': [150, 235], 'masks_untouched': True}
STRUCTURAL = {'vertical_flip': 0.5, 'paste_probability': 0.5, 'patches': [1, 3], 'patch_size': [20, 64], 'lane_x': [120, 224], 'source_x_exclusion': [100, 244], 'blend': 'hard', 'masks_untouched': True}
PHOTOMETRIC = {'hsv_probability': 0.5, 'hue_shift_max': 10, 'saturation_scale': [0.6, 1.4], 'blur_probability': 0.5, 'blur_sigma': [0.3, 1.5], 'downsample_probability': 0.3, 'downsample_factor': [1.2, 2.0], 'noise_probability': 0.3, 'noise_sigma': [2.0, 8.0], 'jpeg_probability': 0.3, 'jpeg_quality': [30, 80]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n');temp.replace(path)


def gpu_idle():
    """Do not overlap another user's or another experiment's GPU work."""
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    if {int(x) for x in p.stdout.split()} - {os.getpid()}:
        raise RuntimeError('GPU has other work; no automatic retry')


def read_pair(root, row):
    """Load exact pixels and the union of every declared conveyor cap instance."""
    image = cv2.imread(str(root/row['path']))
    mask = cv2.imread(str(root/row['mask_path']), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (480, 640, 3) or mask is None or mask.shape != (480, 640) or mask.dtype != np.uint16:
        raise ValueError('Invalid native image or instance mask')
    return image, mask


def augment(image, mask, rng, objects=None):
    """Flip both arrays together; every other transform is photometric and affects RGB only.

    Run45: domain-randomisation extras after the original gamma/gain. Each is drawn from the same
    per-sample RNG so an epoch's draws stay reproducible from (seed, epoch, index).
    """
    flipped = rng.random() < .5
    if flipped:
        image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
    if rng.random() < STRUCTURAL['vertical_flip']:
        image, mask = image[::-1].copy(), mask[::-1].copy()
    if rng.random() < .5:
        gamma = np.exp(rng.uniform(np.log(.4), np.log(2.2)))
        lut = np.clip((np.arange(256)/255.)**gamma*255, 0, 255).astype(np.uint8)
        image = lut[image]
    gain = rng.uniform(.8, 1.2, (1, 1, 3))
    image = np.clip(image.astype(np.float32)*gain, 0, 255).astype(np.uint8)
    if rng.random() < PHOTOMETRIC['hsv_probability']:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.uniform(-PHOTOMETRIC['hue_shift_max'], PHOTOMETRIC['hue_shift_max']))) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(*PHOTOMETRIC['saturation_scale']), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < PHOTOMETRIC['blur_probability']:
        image = cv2.GaussianBlur(image, (0, 0), float(rng.uniform(*PHOTOMETRIC['blur_sigma'])))
    if rng.random() < PHOTOMETRIC['downsample_probability']:
        factor = float(rng.uniform(*PHOTOMETRIC['downsample_factor']))
        small = cv2.resize(image, (max(1, int(round(640/factor))), max(1, int(round(480/factor)))), interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (640, 480), interpolation=cv2.INTER_LINEAR)
    if rng.random() < PHOTOMETRIC['noise_probability']:
        sigma = float(rng.uniform(*PHOTOMETRIC['noise_sigma']))
        image = np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape).astype(np.float32), 0, 255).astype(np.uint8)
    if rng.random() < PHOTOMETRIC['jpeg_probability']:
        quality = int(rng.uniform(*PHOTOMETRIC['jpeg_quality']))
        ok, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError('JPEG re-encoding failed')
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    image = paste_structural_negatives(image, mask, rng, flipped)
    image = draw_procedural_negatives(image, mask, rng, flipped)
    image, mask, objects = paste_caps(image, mask, objects, rng, flipped)
    if image.shape != (480, 640, 3) or image.dtype != np.uint8 or mask.shape != (480, 640) or mask.dtype != np.uint16:
        raise ValueError('Augmentation changed native geometry')
    return image, mask, objects

def paste_structural_negatives(image, mask, rng, flipped=False):
    """Copy non-cap patches from outside the lane band into the lane band; never cover a cap pixel."""
    if rng.random() >= STRUCTURAL['paste_probability']:
        return image
    image = image.copy()
    count = int(rng.integers(STRUCTURAL['patches'][0], STRUCTURAL['patches'][1] + 1))
    for _ in range(count):
        for _attempt in range(10):
            w = int(rng.integers(STRUCTURAL['patch_size'][0], STRUCTURAL['patch_size'][1] + 1))
            h = int(rng.integers(STRUCTURAL['patch_size'][0], STRUCTURAL['patch_size'][1] + 1))
            # source: anywhere outside the lane band, with no cap pixels
            sx = int(rng.integers(0, 640 - w)); sy = int(rng.integers(0, 480 - h))
            if STRUCTURAL['source_x_exclusion'][0] - w < sx < STRUCTURAL['source_x_exclusion'][1]:
                continue
            if mask[sy:sy+h, sx:sx+w].any():
                continue
            # destination: inside the lane band, with no cap pixels under the patch
            lx0, lx1 = (640 - STRUCTURAL['lane_x'][1], 640 - STRUCTURAL['lane_x'][0]) if flipped else STRUCTURAL['lane_x']
            dx = int(rng.integers(lx0, lx1 - w)); dy = int(rng.integers(0, 480 - h))
            if mask[dy:dy+h, dx:dx+w].any():
                continue
            image[dy:dy+h, dx:dx+w] = image[sy:sy+h, sx:sx+w]
            break
    return image


class Frames(Dataset):
    """Keep every original frame and pixel, without replacement or cropping."""

    def __init__(self, root, rows, seed, epoch, training, mean_area=None):
        self.root, self.rows = root, rows
        self.seed, self.epoch, self.training = seed, epoch, training
        self.mean_area = mean_area

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, mask = read_pair(self.root, self.rows[index])
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            image, mask, objects = augment(image, mask, rng, self.rows[index]['objects'])
        result = (image_tensor([image], 'cpu')[0], torch.from_numpy((mask > 0).astype(np.float32)[None]), index)
        if self.training:
            weights = foreground_weights(mask, objects, self.mean_area)
            return (*result, torch.from_numpy(weights[None]))
        return result


def verify(root):
    """Verify all source bindings, full masks, and sequence/pixel split separation."""
    protocol = json.loads((root/'protocol.json').read_text())
    epoch_learning_rate(protocol,0)
    if protocol['real_inputs_used'] is not False or protocol['initialization'] != 'random':
        raise ValueError('Synthetic-only random initialization required')
    for name, digest in protocol['source_sha256'].items():
        if sha(root/'source'/name) != digest:
            raise ValueError('Source changed: '+name)
    dataset = Path(protocol['dataset'])
    if sha(dataset/'manifest.json') != protocol['dataset_sha256']:
        raise ValueError('Dataset manifest changed')
    manifest = json.loads((dataset/'manifest.json').read_text())
    expected_counts = protocol.get('dataset_counts', {'train':4440, 'val':1280})
    if expected_counts not in ({'train':4440, 'val':1280}, {'train':5160, 'val':1280}):
        raise ValueError('Unrecognized frozen dataset population')
    if manifest['real_inputs_used'] is not False or manifest['counts'] != expected_counts:
        raise ValueError('Dataset population differs')
    keys, pixels, role_counts, identities = {}, {}, {'train':0, 'val':0}, set()
    stats = {r:dict(frames=0, instances=0, empty=0, foreground_pixels=0, touching_instance_frames=0) for r in role_counts}
    for row in manifest['records']:
        role = row['role']; key = (row['component'], row['sequence'])
        identity = (*key, row['frame'])
        if role not in role_counts or identity in identities or keys.get(key, role) != role or pixels.get(row['sha256'], role) != role:
            raise ValueError('Duplicate frame or split leakage')
        identities.add(identity); keys[key] = role; pixels[row['sha256']] = role
        for field, digest in [('path','sha256'), ('mask_path','mask_sha256'), ('label_path','label_sha256')]:
            path = (dataset/row[field]).resolve()
            if not path.is_relative_to(dataset.resolve()) or sha(path) != row[digest]:
                raise ValueError('Dataset byte/path mismatch')
        image, mask = read_pair(dataset, row)
        if set(np.unique(mask))-{0} != {o['mask_value'] for o in row['objects']}:
            raise ValueError('Mask target inventory differs')
        for obj in row['objects']:
            ys, xs = np.where(mask == obj['mask_value'])
            if len(xs) != obj['pixels'] or [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] != obj['bbox_xyxy']:
                raise ValueError('Mask/object truth mismatch')
        components, labels = cv2.connectedComponents((mask>0).astype(np.uint8), connectivity=8)
        touching = any(len(np.unique(mask[labels == i])) > 1 for i in range(1, components))
        s=stats[role];s['frames']+=1;s['instances']+=len(row['objects']);s['empty']+=not row['objects']
        s['foreground_pixels']+=int((mask>0).sum());s['touching_instance_frames']+=touching
        role_counts[role] += 1
    if role_counts != manifest['counts']:
        raise ValueError('Actual frame counts differ')
    area = protocol['objective']['foreground_area_weighting']
    if (area['mean_area_numerator'] != stats['train']['foreground_pixels']
            or area['mean_area_denominator'] != stats['train']['instances']):
        raise ValueError('TRAIN-derived mean area differs')
    return protocol, manifest, stats


def run(root, check_only=False):
    """Train once, select only on synthetic validation, and retain exact epoch order."""
    protocol, manifest, audit = verify(root)
    save(root/'data-audit.json', dict(passed=True, dataset_sha256=protocol['dataset_sha256'],
                                     stats=audit, real_inputs_used=False))
    if check_only:
        print('DENSE_FULL_DATA_SOURCE_CHECK_PASS', flush=True)
        return
    if torch.__version__ != protocol['torch_version']:
        raise ValueError('Torch runtime changed')
    proof = json.loads((root/'cpu-proof.json').read_text())
    if (sha(root/'cpu-proof.json') != protocol['cpu_proof_sha256'] or proof['passed'] is not True
            or proof['source_sha256'] != protocol['source_sha256']):
        raise ValueError('CPU preflight changed')
    gpu_idle(); seed=protocol['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    area = protocol['objective']['foreground_area_weighting']
    mean_area = area['mean_area_numerator'] / area['mean_area_denominator']
    train=sorted((r for r in manifest['records'] if r['role']=='train'),key=lambda r:(r['component'],r['sequence'],r['frame']))
    val=sorted((r for r in manifest['records'] if r['role']=='val'),key=lambda r:(r['component'],r['sequence'],r['frame']))
    model=DenseCapNet().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=epoch_learning_rate(protocol,0),weight_decay=.0005)
    out=root/'train01';out.mkdir(exist_ok=False)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(initial,out/'initial.pt')
    rows=[];best=-1.;best_epoch=None;start=time.monotonic();stop='max_epochs';partial=None
    for epoch in range(protocol['epochs']):
        applied_learning_rate=set_epoch_learning_rate(optimizer,protocol,epoch)
        applied_learning_rates=[group['lr'] for group in optimizer.param_groups]
        gpu_idle();model.train();g=torch.Generator().manual_seed(seed+epoch)
        order=torch.randperm(len(train),generator=g).tolist()
        loader=DataLoader(Frames(Path(protocol['dataset']),train,seed,epoch,True,mean_area),
            batch_size=protocol['batch'],sampler=order,num_workers=4,pin_memory=True)
        seen=[];loss_sum=0.
        for images,target,indices,weights in loader:
            if time.monotonic()-start > protocol['budget_seconds']:
                stop='time_budget';partial=dict(epoch=epoch,learning_rate=applied_learning_rate,learning_rates=applied_learning_rates,indices=seen);break
            images=images.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
            weights=weights.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True);loss=instance_balanced_loss(model(images),target,weights)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();seen.extend(indices.tolist());loss_sum+=float(loss.detach())*len(indices)
        if partial is not None:break
        if seen!=order or sorted(seen)!=list(range(protocol.get('dataset_counts', {'train':4440})['train'])):
            raise ValueError('Training consumption differs')
        save(out/f'epoch-{epoch:03d}-draws.json',dict(epoch=epoch,learning_rate=applied_learning_rate,learning_rates=applied_learning_rates,seed=seed+epoch,indices=seen))
        model.eval();dice_sum=0.;intersection=0;union=0;negative_fp_pixels=0;val_seen=[]
        raw_dice_sum=0.;raw_intersection=0;raw_union=0;raw_negative_fp_pixels=0;validation_frames=[]
        with torch.no_grad():
            for images,target,indices in DataLoader(Frames(Path(protocol['dataset']),val,seed,epoch,False),batch_size=protocol['batch'],shuffle=False,num_workers=4):
                logits=model(images.cuda());truth=target.cuda().bool();probability=logits.sigmoid()
                raw=probability>=.5
                raw_ins=(raw&truth).sum((1,2,3));raw_den=raw.sum((1,2,3))+truth.sum((1,2,3))
                raw_dice_sum+=float(torch.where(raw_den>0,2*raw_ins/raw_den,1.).sum())
                raw_intersection+=int(raw_ins.sum());raw_union+=int((raw|truth).sum())
                raw_negative_fp_pixels+=int(raw[~truth.flatten(1).any(1)].sum())
                native=probability.cpu().numpy()[:,0]
                for plane,index in zip(native,indices.tolist(),strict=True):
                    row=val[index];mask=cv2.imread(str(Path(protocol['dataset'])/row['mask_path']),cv2.IMREAD_UNCHANGED)
                    diagnostic=frame_metrics(plane,mask,row['objects'])
                    validation_frames.append(dict(index=index,component=row['component'],sequence=row['sequence'],frame=row['frame'],**diagnostic))
                pred=torch.from_numpy(np.stack([instances(plane)[2].any(0) for plane in native])[:,None]).to(truth.device)
                if not torch.isfinite(logits).all():raise ValueError('Nonfinite validation output')
                ins=(pred&truth).sum((1,2,3));den=pred.sum((1,2,3))+truth.sum((1,2,3))
                dice_sum+=float(torch.where(den>0,2*ins/den,1.).sum())
                intersection+=int(ins.sum());union+=int((pred|truth).sum())
                negative_fp_pixels+=int(pred[~truth.flatten(1).any(1)].sum());val_seen.extend(indices.tolist())
        if val_seen!=list(range(1280)):raise ValueError('Incomplete validation')
        score=dice_sum/len(val)
        diagnostics=summarize(validation_frames)
        if abs(diagnostics['all']['filtered']['mean_frame_dice']-score)>1e-6:
            raise ValueError('Per-frame filtered Dice differs from selection')
        save(out/f'epoch-{epoch:03d}-validation-frames.json',validation_frames)
        save(out/f'epoch-{epoch:03d}-validation-summary.json',diagnostics)
        entry=dict(epoch=epoch,learning_rate=applied_learning_rate,learning_rates=applied_learning_rates,train_loss=loss_sum/len(train),mean_frame_dice=score,
                   global_iou=intersection/union if union else 1.,negative_fp_pixels=negative_fp_pixels,
                   raw_mean_frame_dice=raw_dice_sum/len(val),raw_global_iou=raw_intersection/raw_union if raw_union else 1.,
                   raw_negative_fp_pixels=raw_negative_fp_pixels,minimum_predicted_component_pixels=9,
                   elapsed_seconds=time.monotonic()-start)
        checkpoint=out/f'epoch-{epoch:03d}.pt'
        temporary=checkpoint.with_suffix('.pt.tmp');torch.save(model.state_dict(),temporary);temporary.replace(checkpoint)
        entry['checkpoint_sha256']=sha(checkpoint)
        entry['validation_frames_sha256']=sha(out/f'epoch-{epoch:03d}-validation-frames.json')
        entry['validation_summary_sha256']=sha(out/f'epoch-{epoch:03d}-validation-summary.json')
        rows.append(entry);save(out/'epochs.json',rows);print(json.dumps(entry),flush=True)
        if score>=best:
            best=score;best_epoch=epoch
            temporary=out/'best.pt.tmp';temporary.write_bytes(checkpoint.read_bytes());temporary.replace(out/'best.pt')
        if epoch-best_epoch>=protocol['patience']:
            stop='patience';break
    if partial is not None:save(out/'partial-epoch.json',partial)
    if best_epoch is None:raise RuntimeError('No completed validation epoch within budget')
    save(out/'checkpoint-inventory.json',dict(completed_epochs=len(rows),filtered_best_epoch=best_epoch,
        raw_best_epoch=max(range(len(rows)),key=lambda i:(rows[i]['raw_mean_frame_dice'],i)),
        checkpoints=[dict(epoch=row['epoch'],path=f"epoch-{row['epoch']:03d}.pt",sha256=row['checkpoint_sha256']) for row in rows],
        raw_best_is_diagnostic_only=True))
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in selected.values()):raise ValueError('Invalid saved model')
    changed=[k for k,v in selected.items() if not torch.equal(v,initial[k])]
    if not changed:raise ValueError('No learned parameter changes')
    save(out/'frozen-model.json',dict(name=protocol['name'],family='dense_cap_unet_v1',task='segment',
        initialization='random',pretrained=False,real_inputs_used=False,pretraining_origin='synthetic_only',
        weights=str((out/'best.pt').resolve()),weights_sha256=sha(out/'best.pt'),
        initial_weights_sha256=sha(out/'initial.pt'),protocol_sha256=sha(root/'protocol.json'),
        dataset_sha256=protocol['dataset_sha256'],source_sha256=protocol['source_sha256'],
        selected_epoch=best_epoch,selected_mean_frame_dice=best,completed_epochs=len(rows),
        stop=stop,partial_epoch=partial is not None,changed_parameters=len(changed),
        elapsed_seconds=time.monotonic()-start,
        limits=['Synthetic development selection only; original paired detector/mask/ID gates still required.',
                'Semantic foreground components can merge touching caps; fixed minimum9 predicted pixels, all original truth retained.']))



def draw_procedural_negatives(image, mask, rng, flipped=False):
    """Draw dark polylines and light rectangles inside the lane band (mirrored after a horizontal flip)."""
    if rng.random() >= PROCEDURAL['probability']:
        return image
    image = image.copy(); keep = mask > 0
    x0, x1 = (640 - PROCEDURAL['lane_x'][1], 640 - PROCEDURAL['lane_x'][0]) if flipped else PROCEDURAL['lane_x']
    for _ in range(int(rng.integers(PROCEDURAL['rectangles'][0], PROCEDURAL['rectangles'][1] + 1))):
        w = int(rng.integers(PROCEDURAL['rectangle_size'][0], PROCEDURAL['rectangle_size'][1] + 1))
        h = int(rng.integers(PROCEDURAL['rectangle_size'][0], PROCEDURAL['rectangle_size'][1] + 1))
        rx = int(rng.integers(x0, max(x0 + 1, x1 - w))); ry = int(rng.integers(0, 480 - h))
        value = rng.uniform(PROCEDURAL['rectangle_value'][0], PROCEDURAL['rectangle_value'][1]) * rng.uniform(.9, 1.1, 3)
        region = image[ry:ry+h, rx:rx+w]; region[~keep[ry:ry+h, rx:rx+w]] = np.clip(value, 0, 255).astype(np.uint8)
    canvas = np.zeros((480, 640), np.uint8)
    for _ in range(int(rng.integers(PROCEDURAL['strokes'][0], PROCEDURAL['strokes'][1] + 1))):
        n = int(rng.integers(PROCEDURAL['stroke_points'][0], PROCEDURAL['stroke_points'][1] + 1))
        span = int(rng.integers(PROCEDURAL['stroke_span'][0], PROCEDURAL['stroke_span'][1] + 1))
        cx = int(rng.integers(x0, x1)); cy = int(rng.integers(0, 480))
        points = np.stack([np.clip(cx + rng.uniform(-span, span, n), 0, 639), np.clip(cy + rng.uniform(-span, span, n), 0, 479)], 1).astype(np.int32)
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, 255, int(rng.integers(PROCEDURAL['stroke_thickness'][0], PROCEDURAL['stroke_thickness'][1] + 1)), cv2.LINE_AA)
    stroke = (canvas > 0) & ~keep
    value = rng.uniform(PROCEDURAL['stroke_value'][0], PROCEDURAL['stroke_value'][1]) * rng.uniform(.8, 1.2, 3)
    image[stroke] = np.clip(value, 0, 255).astype(np.uint8)
    return image



def paste_caps(image, mask, objects, rng, flipped=False):
    """Duplicate existing cap instances (pixels and labels) into the lane band; returns updated objects."""
    if not objects or rng.random() >= CAPPASTE['probability']:
        return image, mask, objects
    image = image.copy(); mask = mask.copy(); objects = [dict(o) for o in objects]
    x0, x1 = (640 - CAPPASTE['lane_x'][1], 640 - CAPPASTE['lane_x'][0]) if flipped else CAPPASTE['lane_x']
    next_value = max(o['mask_value'] for o in objects) + 1
    for _ in range(int(rng.integers(CAPPASTE['copies'][0], CAPPASTE['copies'][1] + 1))):
        if next_value > 65535:
            break
        src = objects[int(rng.integers(len(objects)))]
        ys, xs = np.where(mask == src['mask_value'])
        if len(xs) == 0:
            continue
        by1, by2, bx1, bx2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        region = mask[by1:by2, bx1:bx2] == src['mask_value']; crop = image[by1:by2, bx1:bx2]
        scale = float(rng.uniform(HOLDER['scale'][0], HOLDER['scale'][1]))
        h0, w0 = region.shape; h, w = max(2, int(round(h0 * scale))), max(2, int(round(w0 * scale)))
        if w >= x1 - x0 or h >= 480:
            continue
        region = cv2.resize(region.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
        if not region.any():
            continue
        for _attempt in range(10):
            dx = int(rng.integers(x0, x1 - w)); dy = int(rng.integers(0, 480 - h))
            if mask[dy:dy+h, dx:dx+w].any():
                continue
            if rng.random() < HOLDER['holder_probability']:
                side = int(max(h, w) * rng.uniform(HOLDER['holder_size_factor'][0], HOLDER['holder_size_factor'][1]))
                hx = max(0, min(640 - side, dx + w // 2 - side // 2)); hy = max(0, min(480 - side, dy + h // 2 - side // 2))
                if not mask[hy:hy+side, hx:hx+side].any():
                    value = rng.uniform(HOLDER['holder_value'][0], HOLDER['holder_value'][1]) * rng.uniform(.85, 1.15, 3)
                    image[hy:hy+side, hx:hx+side] = np.clip(value, 0, 255).astype(np.uint8)
            image[dy:dy+h, dx:dx+w][region] = crop[region]
            mask[dy:dy+h, dx:dx+w][region] = next_value
            objects.append(dict(mask_value=next_value, pixels=int(region.sum()), instance_id=str(src.get('instance_id', 'cap')) + '_paste'))
            next_value += 1
            break
    return image, mask, objects


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args();run(args.root,args.check_only)
