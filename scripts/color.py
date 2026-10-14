import numpy as np,sys,json,csv,subprocess
from PIL import Image
S=sys.argv[1]; R=S+'/ds/csi-vial-fill-dataset/vial_fill/'
def lab(rgb):
    c=rgb/255.0; c=np.where(c>0.04045,((c+0.055)/1.055)**2.4,c/12.92)
    M=np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]])
    xyz=c@M.T/np.array([0.95047,1,1.08883]); f=np.where(xyz>0.008856,np.cbrt(xyz),7.787*xyz+16/116)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)
vials=list(csv.DictReader(open(R+'labels/video5_vial_recipes.csv')))
res=[]
for v in vials:
    n=int(v['vial']); acc=[]
    for off in (15,30,45):
        fr=int(v['end_frame'])+off
        subprocess.run(['ffmpeg','-v','error','-y','-ss','%.3f'%(fr/30),'-i',R+'videos/vial_fill_video_5.mp4','-frames:v','1',S+'/c.png'])
        a=np.asarray(Image.open(S+'/c.png').convert('RGB'),np.float32)[150:430,180:420]
        mx=a.max(-1);mn=a.min(-1);sat=(mx-mn)/(mx+1)
        m=(sat>np.percentile(sat,99.3))
        acc.append(np.median(a[m],0))
    rgb=np.median(np.array(acc),0); L=lab(rgb)
    res.append({'vial':n,'recipe':int(v['recipe_id']),'rgb':[int(x) for x in rgb],'lab':[round(float(x),1) for x in L]})
# carryover distance: first vial vs last vial in batch
for r in res:
    last=[x for x in res if x['recipe']==r['recipe']][-1]
    r['dE_vs_last']=round(float(np.linalg.norm(np.array(r['lab'])-np.array(last['lab']))),1)
json.dump(res,open(S+'/colors.json','w'))
for r in res: print(r['vial'],r['recipe'],r['rgb'],r['dE_vs_last'])
