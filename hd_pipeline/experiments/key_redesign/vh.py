import sys; sys.path.insert(0,'.')
from insitu import van, hd, nframes
from PIL import Image
out=sys.argv[1]; rows=[]
for n in sys.argv[2:]:
  ims=[]
  for k in range(nframes(n)):
    h=hd(n,k); v=van(n,k); ims += [v.resize(h.size,Image.NEAREST), h]
  rows.append(ims)
W=max(sum(i.width+6 for i in r) for r in rows); H=sum(max(i.height for i in r)+6 for r in rows)
s=Image.new('RGBA',(W,H),(90,70,50,255)); y=0
for r in rows:
  x=0
  for i in r: s.alpha_composite(i,(x,y)); x+=i.width+6
  y+=max(i.height for i in r)+6
s.convert('RGB').save(out); print(s.size)
