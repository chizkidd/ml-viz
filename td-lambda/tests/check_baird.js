const S=require('../src/sim.js'), ref=require(process.argv[2]); let worst=0;
for(const o of ref){const [dist,tab,boot,algo,alpha]=o.cfg;
 const r=S.bairdRun({dist,tab,boot,algo,alpha,steps:400,seed:3}); const w=r.W[r.W.length-1];
 let rel=0; for(let i=0;i<w.length;i++) rel=Math.max(rel,Math.abs(w[i]-o.w[i])/Math.max(1,Math.abs(o.w[i])));
 worst=Math.max(worst,rel); console.log(o.cfg.join(','), 'maxabsv js',r.V[r.V.length-1].toPrecision(6),'py',o.v.toPrecision(6),'relw',rel.toExponential(2));}
console.log('worst rel diff',worst); process.exit(worst<1e-9?0:1);
