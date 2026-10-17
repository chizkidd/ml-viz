import numpy as np, json
from ref import mulberry32
G=0.99; CAP=1e12
def feats(tab):
    if tab: return np.eye(7)
    X=np.zeros((7,8))
    for i in range(6): X[i,i]=2; X[i,7]=1
    X[6,6]=1; X[6,7]=2
    return X
def run(dist,tab,boot,algo,alpha,steps,seed):
    X=feats(tab); d=X.shape[1]; beta=5*alpha
    w=np.ones(d); 
    if not tab: w[6]=10
    vv=np.zeros(d); rng=mulberry32(seed); s=int(np.floor(rng()*7)); blown=False
    for t in range(steps):
        if blown: break
        if dist=='uniform':
            x7=X[6]; v7=x7@w; dw=np.zeros(d); dv=np.zeros(d)
            for i in range(7):
                xi=X[i]; vi=xi@w; tgt=G*v7 if boot else 0.0; delta=tgt-vi
                if algo=='tdc':
                    dw+=(delta*xi-((G*(xi@vv)*x7) if boot else 0))/7
                    dv+=(delta-xi@vv)*xi/7
                else: dw+=delta*xi/7
            w=w+alpha*dw
            if algo=='tdc': vv=vv+beta*dv
        else:
            solid=rng()<1/7
            s2=6 if solid else int(np.floor(rng()*6))
            rho=(7.0 if solid else 0.0) if dist=='off' else 1.0
            xs=X[s]; x2=X[s2]
            delta=(G*(x2@w) if boot else 0.0)-xs@w
            if algo=='tdc':
                xv=xs@vv
                wn=w+alpha*rho*(delta*xs-((G*x2*xv) if boot else 0))
                vv=vv+beta*rho*(delta-xv)*xs; w=wn
            else: w=w+alpha*rho*delta*xs
            s=s2
        if (not np.all(np.isfinite(w))) or np.abs(w).max()>CAP: blown=True; w=np.clip(np.nan_to_num(w,nan=CAP,posinf=CAP,neginf=-CAP),-CAP,CAP)
    return w, np.abs(X@w).max(), blown
cfgs=[('off',False,True,'td',0.01),('off',True,True,'td',0.01),('off',False,False,'td',0.01),('on',False,True,'td',0.01),
      ('off',False,True,'tdc',0.01),('uniform',False,True,'td',0.01),('uniform',False,True,'tdc',0.01),('on',False,True,'tdc',0.01)]
out=[]
for c in cfgs:
    w,v,b=run(*c,steps=400,seed=3); out.append({'cfg':c,'w':list(map(float,w)),'v':float(v)})
json.dump(out,open('ref_baird.json','w')); 
for o in out: print(o['cfg'], f"{o['v']:.6g}")
