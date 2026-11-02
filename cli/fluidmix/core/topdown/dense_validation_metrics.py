"""Full-frame diagnostics; only filtered mean frame Dice selects checkpoints."""
import numpy as np
from score_instance_masks import mask_ious, match_iou
from fluidmix.topdown.segmenter import instances


def frame_metrics(probability, mask, objects):
    """Score unchanged full-image truth for raw and fixed-nine-pixel output."""
    if mask.shape!=(480,640) or mask.dtype!=np.uint16:
        raise ValueError('Native uint16 instance truth required')
    if set(np.unique(mask))-{0}!={o['mask_value'] for o in objects}:
        raise ValueError('Truth inventory differs')
    truth=np.asarray([mask==o['mask_value'] for o in objects],bool).reshape(-1,480,640)
    foreground=mask>0
    tiny={i for i,o in enumerate(objects) if 9<=o['pixels']<50}
    result=dict(truth_pixels=int(foreground.sum()),truth_instances=len(objects),tiny_truth_instances=len(tiny))
    for name,minimum in [('raw',1),('filtered',9)]:
        predicted=instances(probability,minimum)[2];union=predicted.any(0)
        intersection=int((union&foreground).sum());den=int(union.sum()+foreground.sum())
        pairs=match_iou(mask_ious(predicted,truth),.5)
        result[name]=dict(dice=2*intersection/den if den else 1.,
            intersection=intersection,union=int((union|foreground).sum()),
            predicted_pixels=int(union.sum()),TP=len(pairs),FP=len(predicted)-len(pairs),FN=len(objects)-len(pairs),
            tiny_TP=sum(j in tiny for i,j in pairs),tiny_FN=len(tiny)-sum(j in tiny for i,j in pairs))
    return result


def summarize(frames):
    """Retain positive/empty distinctions; do not select on these diagnostics."""
    result={}
    for scope,population in [('all',frames),('positive',[r for r in frames if r['truth_pixels']]),
                             ('empty',[r for r in frames if not r['truth_pixels']])]:
        result[scope]=dict(frames=len(population),tiny_truth_instances=sum(r['tiny_truth_instances'] for r in population))
        for mode in ('raw','filtered'):
            sums={key:sum(r[mode][key] for r in population) for key in
                  ('intersection','union','predicted_pixels','TP','FP','FN','tiny_TP','tiny_FN')}
            sums['mean_frame_dice']=sum(r[mode]['dice'] for r in population)/len(population) if population else None
            result[scope][mode]=sums
    return result
