import csv,sys,datetime as D
S=sys.argv[1];R=S+'/ds/csi-vial-fill-dataset/vial_fill/'
T0=D.datetime(2026,9,23,19,47,11,161000)  # burned-in clock on video-5 frame 0
K=(D.datetime(2026,9,23,20,21,33,830000)-T0).total_seconds()/1697.0  # real s per video s
vials=list(csv.DictReader(open(R+'labels/video5_vial_recipes.csv')))
idx=[r for r in csv.DictReader(open(R+'image_index.csv')) if r['video_id']=='5']
out=[];last={}
for r in idx:
    w=(D.datetime.fromisoformat(r['timestamp'])-T0).total_seconds(); vt=w/K
    hit=next((v for v in vials if float(v['start_s'])<=vt<=float(v['end_s'])),None)
    row={'filename':r['filename'],'timestamp':r['timestamp'],'video_s_corrected':round(vt,3),'in_fill':1 if hit else 0,
         'vial':hit['vial'] if hit else '','recipe_id':hit['recipe_id'] if hit else ''}
    for c in ['type','t100_ml','t200_ml','t400_ml','total_ml','dispense_order']: row[c]=hit[c] if hit else ''
    out.append(row)
    if hit: last[int(hit['vial'])]=r['filename']
with open(S+'/video5_frame_labels_corrected.csv','w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(out[0].keys()));wr.writeheader();wr.writerows(out)
print('K=%.5f  real fps=%.2f'%(K,30/K)); print('frames',len(out),'in_fill',sum(o['in_fill'] for o in out))
print('vials with stills',sorted(last)); print('max video t covered by stills %.1f'%max(o['video_s_corrected'] for o in out))
open(S+'/last_corrected.tsv','w').write('\n'.join(f'{k}\t{v}' for k,v in sorted(last.items())))
