const S=require('../src/sim.js'), ref=require(process.argv[2]);
const data=S.sweepPrepare(20,10,7); let worst=0;
for(const k in ref){const [l,a,kind]=k.split(',');const v=S.sweepCell(data,+l,+a,kind,1.0);
 worst=Math.max(worst,Math.abs(v-ref[k]));console.log(k,v.toFixed(10),ref[k].toFixed(10));}
console.log('max abs diff',worst); process.exit(worst<1e-9?0:1);
