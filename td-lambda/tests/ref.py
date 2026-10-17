# Independent Python reference (numpy), same mulberry32 stream as the JS
import numpy as np, json, sys
M32=0xFFFFFFFF
def mulberry32(a):
    a&=M32
    def f():
        nonlocal a
        a=(a+0x6D2B79F5)&M32
        t=((a^(a>>15))*(1|a))&M32
        t=((t+(((t^(t>>7))*(61|t))&M32))&M32)^t
        return (((t^(t>>14))&M32))/4294967296
    return f
N=19; START=9
TRUE=np.array([(i-START)/10 for i in range(N)])
def episode(rng):
    s=START; st=[s]
    while True:
        s+= 1 if rng()>=0.5 else -1
        if s<0: return st,-1
        if s>=N: return st,1
        st.append(s)
def run_ep(w,ep,lam,alpha,kind):
    st,term=ep; e=np.zeros(N)
    for k,s in enumerate(st):
        last=k==len(st)-1
        v2=0.0 if last else w[st[k+1]]
        r=term if last else 0
        d=r+v2-w[s]
        e*=lam
        if kind=='rep': e[s]=1
        else: e[s]+=1
        w+=alpha*d*e
def cell(data,lam,alpha,kind,cap=1.0):
    tot=0;cnt=0
    for eps in data:
        w=np.zeros(N)
        for ep in eps:
            run_ep(w,ep,lam,alpha,kind)
            x=np.sqrt(np.mean((w-TRUE)**2))
            tot+= min(x,cap) if np.isfinite(x) else cap; cnt+=1
    return tot/cnt
def prep(runs,episodes,seed):
    rng=mulberry32(seed)
    return [[episode(rng) for _ in range(episodes)] for _ in range(runs)]
if __name__=='__main__':
    data=prep(20,10,7)
    out={}
    for lam,alpha,kind in [(0,0.3,'acc'),(0.8,0.2,'acc'),(0.8,0.2,'rep'),(1.0,0.1,'acc'),(1.0,0.6,'acc'),(0.4,0.5,'rep')]:
        out[f"{lam},{alpha},{kind}"]=cell(data,lam,alpha,kind)
    json.dump(out,open('ref_sweep.json','w'),indent=1); print(out)
